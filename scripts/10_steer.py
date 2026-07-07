# Copyright 2026. Research: does adding the representation INDUCE refusal?
"""The bidirectional completion of the causal story.

Ablation (05/07/09) showed *removing* the narration directions (p_ref, p_harm)
does not stop refusal behavior; only the harm *representation* (m, m_|_p) does.
The sufficiency test: *add* each direction to BENIGN prompts and see which one
*induces* refusal behavior.

Prediction if behavior is representation-driven, not narration-driven:
  +m / +m_|_p   -> benign prompts start getting refused (behavior induced)
  +p_ref/+p_harm-> workspace lights up "cannot"/"illegal" (narration induced)
                   but far less actual refusal behavior

We sweep the steering coefficient and, at each, report benign surface-refusal
rate (behavior) alongside the induced workspace refusal-mass (narration).

Usage: uv run python scripts/10_steer.py [--quick]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from jrefusal import (
    SteeringHook,
    collect_residuals,
    mean_diff_directions,
    pullback_directions,
    refusal_mass,
    refusal_token_ids,
)
from jrefusal import data as D
from jrefusal.benchmark import is_refusal
from jrefusal.decompose import project_out
from jrefusal.generate import _format, generate
from jrefusal.model import load

QUICK = "--quick" in sys.argv
# add at a mid band where the harm/refusal representations form (avoids
# over-compounding across all layers)
STEER_LAYERS = [12, 16, 20]
READOUT_LAYERS = [12, 16, 20, 24]
COEFFS = [0.1, 0.2, 0.4] if QUICK else [0.1, 0.2, 0.3, 0.4]
N = 24 if QUICK else 64


def main() -> None:
    t0 = time.time()
    hf_model, tokenizer, lens_model, lens = load()
    ref_ids = refusal_token_ids(tokenizer)
    harm_words = (
        "illegal", "Illegal", "illegally", "unlawful", "illicit", "crime",
        "criminal", "Criminal", "felony", "violence", "violent", "weapon",
        "dangerous", "harmful", "违法", "犯罪", "暴力", "危险", "非法",
    )
    harm_ids = refusal_token_ids(tokenizer, words=harm_words)
    layers = [l for l in lens.source_layers if l >= 8]
    device = hf_model.device

    harmful = D.harmful_prompts(96, seed=1)
    benign = D.harmless_prompts(None, seed=1)
    benign_fit = benign[:96]
    benign_eval = benign[96 : 96 + N]

    p_ref = pullback_directions(lens, lens_model, ref_ids, layers).to(device)
    p_harm = pullback_directions(lens, lens_model, harm_ids, layers).to(device)
    h_res = collect_residuals(lens_model, [_format(tokenizer, x, False) for x in harmful], layers)
    b_res = collect_residuals(lens_model, [_format(tokenizer, x, False) for x in benign_fit], layers)
    m = mean_diff_directions(h_res, b_res).to(device)
    m_perp = project_out(m, p_ref)

    @torch.no_grad()
    def workspace_refusal(prompts):
        vals = []
        for u in prompts:
            ll, _, _ = lens.apply(lens_model, _format(tokenizer, u, False),
                                  layers=READOUT_LAYERS, positions=[-1])
            vals.append(sum(refusal_mass(ll[l][0], ref_ids).item()
                            for l in READOUT_LAYERS) / len(READOUT_LAYERS))
        return sum(vals) / len(vals)

    base_refusal = sum(is_refusal(r) for r in
                       generate(hf_model, tokenizer, benign_eval, max_new_tokens=40)) / N
    print(f"benign baseline surface refusal = {base_refusal:.2f}\n", flush=True)

    dirs = {"p_ref (narration)": p_ref, "p_harm (narration)": p_harm,
            "m_|_p (representation)": m_perp, "m (representation)": m}
    results = {}
    print("steer benign prompts (+coeff*dir at L12/16/20):", flush=True)
    print("  direction                coeff  surface_refusal  workspace_refusal_mass",
          flush=True)
    for name, d in dirs.items():
        for c in COEFFS:
            with SteeringHook(lens_model.layers, d, coeff=c, layers=STEER_LAYERS):
                resp = generate(hf_model, tokenizer, benign_eval, max_new_tokens=40)
                wm = workspace_refusal(benign_eval)
            sr = sum(is_refusal(r) for r in resp) / N
            results[f"{name} @c{c}"] = {"surface_refusal": sr, "workspace_mass": wm,
                                        "example": resp[0][:120]}
            print(f"  {name:24s} {c:<5}  {sr:6.2f}          {wm:6.2f}", flush=True)
            if abs(c - 0.2) < 1e-9:  # sanity-check the text at the key coeff
                print(f"       e.g. {resp[0][:110]!r}", flush=True)

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "steer.json").write_text(json.dumps(
        {"config": {"quick": QUICK, "steer_layers": STEER_LAYERS, "coeffs": COEFFS,
                    "benign_baseline_refusal": base_refusal}, "results": results},
        indent=2))
    print(f"\ninterpretation: if +m/+m_|_p induce surface refusal while "
          f"+p_ref/+p_harm mostly raise workspace_mass without inducing refusal, "
          f"behavior is representation-driven, not narration-driven (bidirectional).")
    print(f"wrote out/steer.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
