# Copyright 2026. Research: is the harmfulness-PERCEPTION feature the lever?
"""The correction (07_nullspace.py) showed behavioral refusal is driven by a
lens-visible harmfulness-perception feature (m⊥p reads illegal/crime/violence),
distinct from the refusal-narration (I-cannot) our first pullback targeted.

Test that directly: build a SECOND pullback from harm-perception tokens,
p_harm = J_lᵀ(g⊙w_harm), and ask:

  (1) geometry — is p_harm aligned with the behavior-carrying m⊥p, and away from
      the narration pullback p_ref?
  (2) readout — does harm-mass track TOPIC (high on harmful AND benign-harmful-
      topic) while refusal-mass tracks the DECISION (high only on harmful)?
  (3) ablation — does ablating p_harm remove behavioral refusal (like m⊥p) where
      p_ref does not, and at what workspace collateral vs abliteration?

If p_harm removes behavior at low collateral, the harm-perception readout is the
surgical behavioral lever — a lens-derived edit targeting the RIGHT feature.

Usage: uv run python scripts/09_perception.py [--quick]
"""

from __future__ import annotations

import contextlib
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
    refusal_mass,
    refusal_token_ids,
)
from jrefusal import data as D
from jrefusal.benchmark import eval_mc, refusal_rate
from jrefusal.decompose import project_out
from jrefusal.generate import _format, generate
from jrefusal.model import load
from jrefusal.preserve import jspace_readout, kl_distortion

# Harmfulness-perception tokens (legality/criminality), from the m⊥p lens image.
HARM_WORDS = (
    "illegal", "Illegal", "illegally", "unlawful", "illicit",
    "crime", "criminal", "Criminal", "felony",
    "violence", "violent", "weapon", "dangerous", "harmful",
    "违法", "犯罪", "暴力", "危险", "非法",
)

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8
READOUT_LAYERS = [12, 16, 20, 24]
if QUICK:
    N_FIT, N_ADV, N_MC, N_CTRL = 24, 32, 40, 16
    STRENGTHS = [1.0]
else:
    N_FIT, N_ADV, N_MC, N_CTRL = 96, 100, 200, 48
    STRENGTHS = [1.0, 1.5]


def fmt(tok, ps):
    return [_format(tok, p, False) for p in ps]


def main() -> None:
    t0 = time.time()
    hf_model, tokenizer, lens_model, lens = load()
    ref_ids = refusal_token_ids(tokenizer)
    harm_ids = refusal_token_ids(tokenizer, words=HARM_WORDS)
    layers = [l for l in lens.source_layers if l >= ABLATE_FROM]
    device = hf_model.device
    print(f"refusal tokens={len(ref_ids)}  harm tokens={len(harm_ids)} "
          f"(overlap={len(set(ref_ids)&set(harm_ids))}) | quick={QUICK}", flush=True)

    harmful = D.harmful_prompts(None, seed=1)
    harmless = D.harmless_prompts(None, seed=1)
    adv_eval = harmful[N_FIT : N_FIT + N_ADV]
    ctrl = harmless[N_FIT : N_FIT + N_CTRL]
    mc = D.capability_mc(N_MC, seed=2)
    xs = D.xstest(None, seed=2)
    xstest_safe = [p for p, lab in xs if lab == "safe"][:N_CTRL]

    # directions
    p_ref = pullback_directions(lens, lens_model, ref_ids, layers).to(device)
    p_harm = pullback_directions(lens, lens_model, harm_ids, layers).to(device)
    h_res = collect_residuals(lens_model, fmt(tokenizer, harmful[:N_FIT]), layers)
    b_res = collect_residuals(lens_model, fmt(tokenizer, harmless[:N_FIT]), layers)
    m = mean_diff_directions(h_res, b_res).to(device)
    m_perp = project_out(m, p_ref)

    # (1) geometry
    def cos_avg(a, b):
        c = a.cosine_to(b)
        return sum(c.values()) / len(c)
    print("\n(1) geometry (avg |cos| across layers):", flush=True)
    print(f"    cos(p_harm, m_|_p)   = {cos_avg(p_harm, m_perp):+.3f}   "
          f"(behavior-carrying dir)", flush=True)
    print(f"    cos(p_harm, m)       = {cos_avg(p_harm, m):+.3f}", flush=True)
    print(f"    cos(p_harm, p_ref)   = {cos_avg(p_harm, p_ref):+.3f}   "
          f"(narration dir)", flush=True)
    print(f"    cos(p_ref,  m_|_p)   = {cos_avg(p_ref, m_perp):+.3f}", flush=True)

    # (2) readout: harm-mass (topic) vs refusal-mass (decision)
    @torch.no_grad()
    def masses(prompts):
        rm, hm = [], []
        for u in prompts:
            ll, _, _ = lens.apply(lens_model, _format(tokenizer, u, False),
                                  layers=READOUT_LAYERS, positions=[-1])
            rm.append(sum(refusal_mass(ll[l][0], ref_ids).item() for l in READOUT_LAYERS) / len(READOUT_LAYERS))
            hm.append(sum(refusal_mass(ll[l][0], harm_ids).item() for l in READOUT_LAYERS) / len(READOUT_LAYERS))
        return sum(rm) / len(rm), sum(hm) / len(hm)
    print("\n(2) readout — refusal-mass (decision) vs harm-mass (topic):", flush=True)
    for name, ps in (("harmful", adv_eval[:N_CTRL]), ("xstest_safe(topic)", xstest_safe),
                     ("benign", ctrl)):
        r, h = masses(ps)
        print(f"    {name:20s} refusal_mass={r:6.2f}   harm_mass={h:6.2f}", flush=True)

    # (3) ablation comparison
    ctrl_fmt = fmt(tokenizer, ctrl)
    before = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
    excl = sorted(set(ref_ids) | set(harm_ids))

    def run(dirs, s):
        cm = contextlib.nullcontext() if dirs is None else AblationHook(lens_model.layers, dirs, strength=s)
        with cm:
            adv = generate(hf_model, tokenizer, adv_eval, max_new_tokens=48)
            acc = eval_mc(hf_model, tokenizer, mc)
            after = jspace_readout(lens, lens_model, ctrl_fmt, READOUT_LAYERS)
        return {"adv_refusal": refusal_rate(adv), "arc_acc": acc,
                "workspace_kl": kl_distortion(before, after, exclude_ids=excl)}

    print("\n(3) ablation (adv_refusal / ARC / workspace_kl off refusal+harm axes):",
          flush=True)
    results = {"original": run(None, 0)}
    print(f"    {'original':28s} {results['original']}", flush=True)
    conds = [("p_ref (narration)", p_ref), ("p_harm (perception)", p_harm),
             ("m (abliteration)", m), ("m_|_p", m_perp)]
    for s in STRENGTHS:
        for name, dirs in conds:
            key = f"{name} @s{s}"
            results[key] = run(dirs, s)
            print(f"    {key:28s} {results[key]}", flush=True)

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "perception.json").write_text(json.dumps({
        "config": {"quick": QUICK, "harm_words": list(HARM_WORDS)},
        "geometry": {
            "cos_pharm_mperp": cos_avg(p_harm, m_perp),
            "cos_pharm_m": cos_avg(p_harm, m),
            "cos_pharm_pref": cos_avg(p_harm, p_ref),
            "cos_pref_mperp": cos_avg(p_ref, m_perp),
        },
        "ablation": results,
    }, indent=2))
    print(f"\nwrote out/perception.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
