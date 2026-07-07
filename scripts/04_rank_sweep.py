# Copyright 2026. Research: how much of the refusal subspace lives in the
# verbalizable workspace?
"""Rank sweep of the pullback subspace.

The single pullback direction plateaus (~1/3 of AdvBench refusal removed before
the model breaks). Does capturing *more* of the refusal representation — a
higher-rank pullback subspace — remove the behavioral refusal the single
direction misses? And does the workspace collateral then converge to
abliteration's?

If high-rank pullback reaches abliteration-level removal only at
abliteration-level collateral, the conclusion is that the behaviorally-decisive
part of refusal is not cleanly separable within the workspace.

Writes out/rank_sweep.json.

Usage:
    uv run python scripts/04_rank_sweep.py [--quick]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from jrefusal import (
    AblationHook,
    pullback_subspace,
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
RANKS = [1, 2, 4, 8, 16, 32]
STRENGTH = 1.0

if QUICK:
    N_ADV, N_MC, N_CTRL = 32, 40, 16
    RANKS = [1, 4, 16]
else:
    N_ADV, N_MC, N_CTRL = 80, 150, 48


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
    adv_eval = harmful[96 : 96 + N_ADV]
    ctrl = harmless[96 : 96 + N_CTRL]
    mc = D.capability_mc(N_MC, seed=2)
    ctrl_fmt = fmt(tokenizer, ctrl)
    harm_fmt = fmt(tokenizer, adv_eval[:N_CTRL])
    print(f"rank sweep {RANKS} @ strength {STRENGTH} | adv={len(adv_eval)} "
          f"mc={len(mc)} ctrl={len(ctrl)} | quick={QUICK}", flush=True)

    before_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
    before_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)

    points: list[dict] = []
    for rank in RANKS:
        dirs = pullback_subspace(lens, lens_model, ref_ids, layers, rank=rank).to(device)
        with AblationHook(lens_model.layers, dirs, strength=STRENGTH):
            adv = generate(hf_model, tokenizer, adv_eval, max_new_tokens=48)
            acc = eval_mc(hf_model, tokenizer, mc)
            after_ctrl = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
            after_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)
        pt = {
            "rank": rank,
            "adv_refusal": refusal_rate(adv),
            "arc_acc": acc,
            "workspace_kl_offaxis": kl_distortion(before_ctrl, after_ctrl,
                                                  exclude_ids=ref_ids),
            "refusal_suppression": refusal_suppression(before_harm, after_harm,
                                                       ref_ids),
        }
        points.append(pt)
        print(f"  rank={rank:<3} adv_ref={pt['adv_refusal']:.3f} "
              f"arc={pt['arc_acc']:.3f} wsKL={pt['workspace_kl_offaxis']:.4f} "
              f"supp={pt['refusal_suppression']:.2f}", flush=True)

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "rank_sweep.json").write_text(json.dumps(
        {"config": {"quick": QUICK, "ranks": RANKS, "strength": STRENGTH,
                    "ablate_from": ABLATE_FROM}, "points": points}, indent=2))
    print(f"\nwrote out/rank_sweep.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
