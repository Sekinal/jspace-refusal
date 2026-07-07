# Copyright 2026. Research: refusal-intent monitor from the J-space.
"""When a model is made to comply, does its workspace still flag harm?

Two ways to make Qwen comply on harmful requests:

  (a) prompt jailbreaks — standard wrappers. Qwen3.5-4B is quite robust to
      these, so few succeed; we report the table for completeness.
  (b) abliteration — ablate the *automatic* refusal direction (m ⊥ p). From the
      decomposition this removes behavioral refusal (~0.09) while barely
      touching the verbalizable workspace disposition (suppression ~1.8). This
      is the realistic "uncensored open model" setting, and it gives real n.

For the abliterated model we read the workspace refusal-mass (hook active, i.e.
what a deployed monitor would see) and ask: does it still separate
harmful-that-complied from benign? Detection AUC is the headline — the internal
refusal signal survives abliteration and is monitorable.

Usage: uv run python scripts/06_monitor.py [--quick]
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
    refusal_mass,
    refusal_token_ids,
)
from jrefusal import data as D
from jrefusal.benchmark import is_refusal
from jrefusal.decompose import project_out
from jrefusal.generate import _format, generate
from jrefusal.jailbreak import JAILBREAK_TEMPLATES, apply_jailbreaks
from jrefusal.model import load

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8
READOUT_LAYERS = [16, 20, 24]
if QUICK:
    N_FIT, N_HARM, N_BENIGN = 24, 24, 24
else:
    N_FIT, N_HARM, N_BENIGN = 96, 64, 96


def auc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    t0 = time.time()
    hf_model, tokenizer, lens_model, lens = load()
    ref_ids = refusal_token_ids(tokenizer)
    layers = [l for l in lens.source_layers if l >= ABLATE_FROM]

    harmful = D.harmful_prompts(None, seed=7)
    benign = D.harmless_prompts(None, seed=7)
    harmful_fit = harmful[:N_FIT]
    harm_eval = harmful[N_FIT : N_FIT + N_HARM]
    benign_eval = benign[:N_BENIGN]

    @torch.no_grad()
    def workspace_mass(user_prompts: list[str]) -> list[float]:
        """Mean refusal-mass over READOUT_LAYERS at the generation position,
        reflecting any active hook (what a deployed monitor would read)."""
        vals = []
        for u in user_prompts:
            text = _format(tokenizer, u, False)
            ll, _, _ = lens.apply(lens_model, text, layers=READOUT_LAYERS, positions=[-1])
            vals.append(sum(refusal_mass(ll[l][0], ref_ids).item()
                            for l in READOUT_LAYERS) / len(READOUT_LAYERS))
        return vals

    # ---------- (a) prompt jailbreaks ----------
    wrapped = apply_jailbreaks(harm_eval)
    jb_resp = generate(hf_model, tokenizer, [w[2] for w in wrapped], max_new_tokens=48)
    jb_refused = [is_refusal(r) for r in jb_resp]
    print(f"\n(a) prompt jailbreaks: {len(harm_eval)} reqs x "
          f"{len(JAILBREAK_TEMPLATES)} templates", flush=True)
    print("  template          surface_refusal", flush=True)
    for name in JAILBREAK_TEMPLATES:
        idx = [i for i, w in enumerate(wrapped) if w[0] == name]
        print(f"    {name:15s}  {sum(jb_refused[i] for i in idx)/len(idx):.2f}",
              flush=True)

    # ---------- (b) abliteration jailbreak ----------
    p = pullback_directions(lens, lens_model, ref_ids, layers).to(hf_model.device)
    h_res = collect_residuals(lens_model, [_format(tokenizer, x, False) for x in harmful_fit], layers)
    b_res = collect_residuals(lens_model, [_format(tokenizer, x, False) for x in benign[:N_FIT]], layers)
    m = mean_diff_directions(h_res, b_res).to(hf_model.device)
    auto = project_out(m, p)  # automatic-only direction

    with AblationHook(lens_model.layers, auto):
        abl_resp = generate(hf_model, tokenizer, harm_eval, max_new_tokens=48)
        abl_mass = workspace_mass(harm_eval)      # workspace read on the deployed (abliterated) model
        benign_mass = workspace_mass(benign_eval)
    abl_refused = [is_refusal(r) for r in abl_resp]

    complied = [i for i in range(len(harm_eval)) if not abl_refused[i]]
    print(f"\n(b) abliteration (automatic m_|_p): surface refusal on harmful = "
          f"{sum(abl_refused)/len(harm_eval):.2f}  ({len(complied)}/{len(harm_eval)} complied)",
          flush=True)
    print(f"    mean workspace refusal-mass: harmful(complied)="
          f"{sum(abl_mass[i] for i in complied)/max(1,len(complied)):.2f}  "
          f"benign={sum(benign_mass)/len(benign_mass):.2f}", flush=True)

    # detection: does the workspace separate harmful-that-complied from benign?
    sub_mass = [abl_mass[i] for i in complied] + benign_mass
    sub_labels = [1] * len(complied) + [0] * len(benign_mass)
    sub_surf = [0.0] * len(complied) + [1.0 if is_refusal(r) else 0.0
                                        for r in generate(hf_model, tokenizer, benign_eval, max_new_tokens=8)]
    auc_ws = auc(sub_mass, sub_labels)
    auc_surf = auc(sub_surf, sub_labels)
    print(f"\n=== detection AUC on the abliterated model (harmful-complied vs benign) ===")
    print(f"    workspace refusal-mass:  {auc_ws:.3f}")
    print(f"    surface behavior:        {auc_surf:.3f}", flush=True)
    print(f"\ninterpretation: abliteration removes surface refusal (AUC ~0.5) but the"
          f" workspace still flags harm (AUC ~{auc_ws:.2f}) -> monitorable from the inside.")

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "monitor.json").write_text(json.dumps({
        "config": {"quick": QUICK, "readout_layers": READOUT_LAYERS},
        "prompt_jailbreak_surface_refusal": {
            name: sum(jb_refused[i] for i, w in enumerate(wrapped) if w[0] == name)
                  / max(1, sum(w[0] == name for w in wrapped))
            for name in JAILBREAK_TEMPLATES},
        "abliteration": {
            "surface_refusal": sum(abl_refused) / len(harm_eval),
            "n_complied": len(complied),
            "workspace_mass_harmful": sum(abl_mass[i] for i in complied) / max(1, len(complied)),
            "workspace_mass_benign": sum(benign_mass) / len(benign_mass),
            "auc_workspace": auc_ws, "auc_surface": auc_surf,
        },
    }, indent=2))
    print(f"\nwrote out/monitor.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
