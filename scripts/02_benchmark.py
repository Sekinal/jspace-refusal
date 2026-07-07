# Copyright 2026. Research benchmark.
"""Original vs. edited, and Jacobian-pullback vs. standard abliteration.

Builds refusal directions three ways on a *fit* split, then evaluates each on a
disjoint *eval* split across four behavioral axes plus the workspace-collateral
metric:

  * AdvBench refusal rate          (efficacy: want DOWN)
  * XSTest unsafe / safe refusal   (want unsafe DOWN, safe stays low)
  * ARC-Easy accuracy              (capability: want UNCHANGED)
  * workspace KL off refusal axis  (collateral on benign controls: want ~0)
  * refusal suppression in J-space (sanity: want UP)

The novel claim is the last two: pullback should remove refusal at *lower*
workspace collateral than mean-diff abliteration.

Usage:
    uv run python scripts/02_benchmark.py [--quick]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from jrefusal import (
    AblationHook,
    collect_residuals,
    mean_diff_directions,
    pullback_directions,
    pullback_subspace,
    refusal_token_ids,
)
from jrefusal import data as D
from jrefusal.benchmark import eval_mc, refusal_rate, score_xstest
from jrefusal.generate import _format, generate
from jrefusal.model import load
from jrefusal.preserve import jspace_readout, kl_distortion, refusal_suppression

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8                       # ablate every fitted layer >= this
READOUT_LAYERS = [12, 16, 20, 24]     # where refusal lives (from the probe)
MAX_NEW = 64

if QUICK:
    N_FIT, N_ADV, N_XS, N_MC, N_CTRL = 24, 24, 20, 40, 16
else:
    N_FIT, N_ADV, N_XS, N_MC, N_CTRL = 96, 120, 200, 250, 48


def fmt(tokenizer, prompts):
    return [_format(tokenizer, p, False) for p in prompts]


def main() -> None:
    t0 = time.time()
    hf_model, tokenizer, lens_model, lens = load()
    ref_ids = refusal_token_ids(tokenizer)
    layers = [l for l in lens.source_layers if l >= ABLATE_FROM]
    device = hf_model.device
    print(f"L={lens_model.n_layers} d={lens_model.d_model} | ablate {len(layers)} "
          f"layers >= {ABLATE_FROM} | {len(ref_ids)} refusal ids | quick={QUICK}",
          flush=True)

    # --- splits (disjoint fit / eval) ---
    harmful_all = D.harmful_prompts(None, seed=1)
    harmless_all = D.harmless_prompts(None, seed=1)
    harmful_fit = harmful_all[:N_FIT]
    harmless_fit = harmless_all[:N_FIT]
    harmful_eval = harmful_all[N_FIT : N_FIT + N_ADV]
    harmless_eval = harmless_all[N_FIT : N_FIT + N_CTRL]
    xs = D.xstest(N_XS, seed=2)
    mc = D.capability_mc(N_MC, seed=2)
    print(f"fit {len(harmful_fit)}h/{len(harmless_fit)}b | eval adv={len(harmful_eval)}"
          f" xstest={len(xs)} mc={len(mc)} ctrl={len(harmless_eval)}", flush=True)

    # --- build directions ---
    print("building directions ...", flush=True)
    pb = pullback_directions(lens, lens_model, ref_ids, layers).to(device)
    pbs = pullback_subspace(lens, lens_model, ref_ids, layers, rank=3).to(device)
    h_res = collect_residuals(lens_model, fmt(tokenizer, harmful_fit), layers)
    b_res = collect_residuals(lens_model, fmt(tokenizer, harmless_fit), layers)
    md = mean_diff_directions(h_res, b_res).to(device)
    cos = pb.cosine_to(md)
    print("cos(pullback, mean_diff) per layer:",
          {l: round(cos[l], 2) for l in sorted(cos)}, flush=True)

    methods = {"mean_diff": md, "pullback": pb, "pullback_subspace(r=3)": pbs}

    # --- workspace readouts (before, shared) ---
    ctrl_fmt = fmt(tokenizer, harmless_eval)
    harm_fmt = fmt(tokenizer, harmful_eval[: N_CTRL])
    before_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
    before_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)

    results: dict[str, dict] = {}

    def run_eval(name: str, hook_cm) -> dict:
        with hook_cm:
            adv = generate(hf_model, tokenizer, harmful_eval, max_new_tokens=MAX_NEW)
            xs_resp = generate(hf_model, tokenizer, [p for p, _ in xs],
                               max_new_tokens=MAX_NEW)
            acc = eval_mc(hf_model, tokenizer, mc)
            after_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
            after_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)
        xr = score_xstest(xs, xs_resp)
        return {
            "adv_refusal": refusal_rate(adv),
            "xstest_unsafe_refusal": xr.unsafe_refusal,
            "xstest_safe_refusal": xr.safe_refusal,
            "arc_acc": acc,
            "workspace_kl_offaxis": kl_distortion(before_ctrl, after_ctrl,
                                                  exclude_ids=ref_ids),
            "refusal_suppression": refusal_suppression(before_harm, after_harm,
                                                       ref_ids),
        }

    import contextlib
    print("\n== ORIGINAL ==", flush=True)
    results["original"] = run_eval("original", contextlib.nullcontext())
    print(results["original"], flush=True)

    for name, dirs in methods.items():
        print(f"\n== {name} ==", flush=True)
        results[name] = run_eval(name, AblationHook(lens_model.layers, dirs))
        print(results[name], flush=True)

    # --- table ---
    cols = ["adv_refusal", "xstest_unsafe_refusal", "xstest_safe_refusal",
            "arc_acc", "workspace_kl_offaxis", "refusal_suppression"]
    hdr = ["method".ljust(22)] + [c[:14].rjust(15) for c in cols]
    print("\n" + " ".join(hdr))
    for name, r in results.items():
        row = [name.ljust(22)] + [f"{r[c]:.3f}".rjust(15) for c in cols]
        print(" ".join(row))

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(
        {"config": {"quick": QUICK, "ablate_from": ABLATE_FROM,
                    "readout_layers": READOUT_LAYERS, "cos_pb_md": cos},
         "results": results}, indent=2))
    print(f"\nwrote out/benchmark.json | {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
