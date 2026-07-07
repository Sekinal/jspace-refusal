# Copyright 2026. Research: refusal-removal vs. workspace-collateral tradeoff.
"""Strength sweep -> Pareto frontier.

For pullback and mean-diff abliteration, sweep the ablation coefficient and, at
each point, measure how much refusal was removed (AdvBench) against how much
collateral the edit caused (off-refusal-axis workspace KL on benign controls,
and ARC capability). The novel claim is a Pareto statement: at matched refusal
removal, the Jacobian-pullback edit costs less workspace collateral.

Writes out/tradeoff.json (all points) and prints a per-method curve.

Usage:
    uv run python scripts/03_tradeoff.py [--quick]
"""

from __future__ import annotations

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
from jrefusal.generate import _format, generate
from jrefusal.model import load
from jrefusal.preserve import jspace_readout, kl_distortion, refusal_suppression

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8
READOUT_LAYERS = [12, 16, 20, 24]
STRENGTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

if QUICK:
    N_FIT, N_ADV, N_MC, N_CTRL = 24, 32, 40, 16
    STRENGTHS = [0.0, 1.0, 2.0]
else:
    N_FIT, N_ADV, N_MC, N_CTRL = 96, 80, 150, 48


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
    print(f"sweep strengths={STRENGTHS} | adv={len(adv_eval)} mc={len(mc)} "
          f"ctrl={len(ctrl)} | quick={QUICK}", flush=True)

    pb = pullback_directions(lens, lens_model, ref_ids, layers).to(device)
    h_res = collect_residuals(lens_model, fmt(tokenizer, harmful_fit), layers)
    b_res = collect_residuals(lens_model, fmt(tokenizer, harmless_fit), layers)
    md = mean_diff_directions(h_res, b_res).to(device)

    before_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
    before_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)

    points: list[dict] = []
    for method_name, dirs in (("pullback", pb), ("mean_diff", md)):
        for s in STRENGTHS:
            with AblationHook(lens_model.layers, dirs, strength=s):
                adv = generate(hf_model, tokenizer, adv_eval, max_new_tokens=48)
                acc = eval_mc(hf_model, tokenizer, mc)
                after_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
                after_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)
            pt = {
                "method": method_name, "strength": s,
                "adv_refusal": refusal_rate(adv),
                "arc_acc": acc,
                "workspace_kl_offaxis": kl_distortion(before_ctrl, after_ctrl,
                                                      exclude_ids=ref_ids),
                "refusal_suppression": refusal_suppression(before_harm, after_harm,
                                                           ref_ids),
            }
            points.append(pt)
            print(f"  {method_name:10s} s={s:<4} adv_ref={pt['adv_refusal']:.3f} "
                  f"arc={pt['arc_acc']:.3f} wsKL={pt['workspace_kl_offaxis']:.4f} "
                  f"supp={pt['refusal_suppression']:.2f}", flush=True)

    # Pareto summary: min workspace-KL achieving adv_refusal <= target, per method.
    for target in (0.5, 0.2, 0.1):
        print(f"\n-- to reach adv_refusal <= {target} (min workspace collateral) --")
        for m in ("pullback", "mean_diff"):
            cand = [p for p in points if p["method"] == m
                    and p["adv_refusal"] <= target]
            if cand:
                best = min(cand, key=lambda p: p["workspace_kl_offaxis"])
                print(f"   {m:10s} s={best['strength']:<4} "
                      f"wsKL={best['workspace_kl_offaxis']:.4f} "
                      f"arc={best['arc_acc']:.3f}")
            else:
                print(f"   {m:10s} never reaches it in the sweep")

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "tradeoff.json").write_text(json.dumps(
        {"config": {"quick": QUICK, "strengths": STRENGTHS,
                    "ablate_from": ABLATE_FROM}, "points": points}, indent=2))
    print(f"\nwrote out/tradeoff.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
