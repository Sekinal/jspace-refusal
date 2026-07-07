# Copyright 2026. Research: where does behavioral refusal live?
"""Decompose abliteration's direction into a workspace part and an automatic
part, and ablate each to localize behavioral refusal.

Methods evaluated (all strength 1, layers >= 8):
  original         no edit
  pullback         p        — refusal in the verbalizable workspace
  mean_diff        m        — full abliteration direction
  workspace_part   m ∥ p    — the part of abliteration inside the workspace
  automatic_only   m ⊥ p    — abliteration with the workspace part removed
  hybrid           span(p, m⊥p)

If automatic_only removes most behavioral refusal and pullback little, refusal
is mostly automatic. If hybrid matches mean_diff removal at lower workspace KL,
the targeted combination beats blanket abliteration.

Usage: uv run python scripts/05_decompose.py [--quick]
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

from jrefusal import (
    AblationHook,
    collect_residuals,
    mean_diff_directions,
    pullback_directions,
    refusal_token_ids,
)
from jrefusal import data as D
from jrefusal.benchmark import eval_mc, refusal_rate
from jrefusal.decompose import parallel_component, project_out, union
from jrefusal.generate import _format, generate
from jrefusal.model import load
from jrefusal.preserve import jspace_readout, kl_distortion, refusal_suppression

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8
READOUT_LAYERS = [12, 16, 20, 24]
MAX_NEW = 48

if QUICK:
    N_FIT, N_ADV, N_MC, N_CTRL = 24, 32, 40, 16
else:
    N_FIT, N_ADV, N_MC, N_CTRL = 96, 100, 200, 48


def fmt(tok, prompts):
    return [_format(tok, p, False) for p in prompts]


def main() -> None:
    t0 = time.time()
    hf_model, tokenizer, lens_model, lens = load()
    ref_ids = refusal_token_ids(tokenizer)
    layers = [l for l in lens.source_layers if l >= ABLATE_FROM]
    device = hf_model.device

    harmful = D.harmful_prompts(None, seed=1)
    harmless = D.harmless_prompts(None, seed=1)
    harmful_fit, harmless_fit = harmful[:N_FIT], harmless[:N_FIT]
    adv_eval = harmful[N_FIT : N_FIT + N_ADV]
    ctrl = harmless[N_FIT : N_FIT + N_CTRL]
    mc = D.capability_mc(N_MC, seed=2)
    ctrl_fmt = fmt(tokenizer, ctrl)
    harm_fmt = fmt(tokenizer, adv_eval[:N_CTRL])
    print(f"decompose | adv={len(adv_eval)} mc={len(mc)} ctrl={len(ctrl)} "
          f"| quick={QUICK}", flush=True)

    # base directions
    p = pullback_directions(lens, lens_model, ref_ids, layers).to(device)
    h_res = collect_residuals(lens_model, fmt(tokenizer, harmful_fit), layers)
    b_res = collect_residuals(lens_model, fmt(tokenizer, harmless_fit), layers)
    m = mean_diff_directions(h_res, b_res).to(device)
    # decomposition
    m_par = parallel_component(m, p)      # workspace part of abliteration
    m_perp = project_out(m, p)            # automatic-only
    hybrid = union(p, m_perp)             # span(p, m⊥p)

    cos = p.cosine_to(m)
    print("cos(pullback, mean_diff):",
          {l: round(cos[l], 2) for l in sorted(cos)[::4]}, flush=True)

    methods = {
        "pullback": p,
        "mean_diff": m,
        "workspace_part(m||p)": m_par,
        "automatic_only(m_|_p)": m_perp,
        "hybrid(p+m_|_p)": hybrid,
    }

    before_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
    before_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)

    def run(hook_cm) -> dict:
        with hook_cm:
            adv = generate(hf_model, tokenizer, adv_eval, max_new_tokens=MAX_NEW)
            acc = eval_mc(hf_model, tokenizer, mc)
            after_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
            after_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)
        return {
            "adv_refusal": refusal_rate(adv),
            "arc_acc": acc,
            "workspace_kl_offaxis": kl_distortion(before_ctrl, after_ctrl,
                                                  exclude_ids=ref_ids),
            "refusal_suppression": refusal_suppression(before_harm, after_harm,
                                                       ref_ids),
        }

    results = {"original": run(contextlib.nullcontext())}
    print("original", results["original"], flush=True)
    for name, dirs in methods.items():
        results[name] = run(AblationHook(lens_model.layers, dirs))
        print(name, results[name], flush=True)

    cols = ["adv_refusal", "arc_acc", "workspace_kl_offaxis", "refusal_suppression"]
    print("\n" + "method".ljust(24) + "".join(c[:15].rjust(17) for c in cols))
    for name, r in results.items():
        print(name.ljust(24) + "".join(f"{r[c]:.3f}".rjust(17) for c in cols))

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "decompose.json").write_text(json.dumps(
        {"config": {"quick": QUICK, "ablate_from": ABLATE_FROM},
         "cos_pb_md": cos, "results": results}, indent=2))
    print(f"\nwrote out/decompose.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
