# Copyright 2026. Research: does the workspace monitor track refusal, or topic?
"""B.4 — the control the monitor was missing.

The AUC-0.998 monitor separated harmful-that-complied from *benign-neutral*.
But refusal-mass on refusal tokens could be high whenever the *topic* is
harmful, regardless of refusal disposition. The discriminating group is
benign-but-harmful-topic: XSTest-safe prompts ("how do I kill a Python
process") that the model does NOT refuse but that sound harmful.

If XSTest-safe clusters with benign -> the workspace tracks refusal disposition
(clean detector). If it clusters with harmful -> partly a topic detector.

Read workspace refusal-mass (mean over layers 16-24, generation position) for
three groups, on the base model and on the automatic-ablated (deployed) model.

Usage: uv run python scripts/08_monitor_control.py [--quick]
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
from jrefusal.benchmark import is_refusal
from jrefusal.decompose import project_out
from jrefusal.generate import _format, generate
from jrefusal.model import load

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8
READOUT_LAYERS = [16, 20, 24]
N = 24 if QUICK else 64


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

    harmful = D.harmful_prompts(N, seed=7)
    benign = D.harmless_prompts(N, seed=7)
    xs = D.xstest(None, seed=2)
    xstest_safe = [p for p, lab in xs if lab == "safe"][:N]   # benign, harmful-topic
    print(f"harmful={len(harmful)} benign={len(benign)} "
          f"xstest_safe={len(xstest_safe)} | quick={QUICK}", flush=True)

    @torch.no_grad()
    def workspace_mass(prompts):
        vals = []
        for u in prompts:
            text = _format(tokenizer, u, False)
            ll, _, _ = lens.apply(lens_model, text, layers=READOUT_LAYERS, positions=[-1])
            vals.append(sum(refusal_mass(ll[l][0], ref_ids).item()
                            for l in READOUT_LAYERS) / len(READOUT_LAYERS))
        return vals

    # automatic-ablation direction (makes the model comply)
    p = pullback_directions(lens, lens_model, ref_ids, layers).to(hf_model.device)
    h_res = collect_residuals(lens_model, [_format(tokenizer, x, False) for x in harmful], layers)
    b_res = collect_residuals(lens_model, [_format(tokenizer, x, False) for x in benign], layers)
    m = mean_diff_directions(h_res, b_res).to(hf_model.device)
    auto = project_out(m, p)

    groups = {"harmful": harmful, "benign_neutral": benign,
              "benign_harmful_topic": xstest_safe}
    report = {}
    for setting, hook in (("base_model", contextlib.nullcontext()),
                          ("abliterated", AblationHook(lens_model.layers, auto))):
        with hook:
            mass = {g: workspace_mass(ps) for g, ps in groups.items()}
            surf = {g: [is_refusal(r) for r in
                        generate(hf_model, tokenizer, ps, max_new_tokens=32)]
                    for g, ps in groups.items()}
        means = {g: sum(v) / len(v) for g, v in mass.items()}
        srate = {g: sum(v) / len(v) for g, v in surf.items()}
        # AUCs: harmful vs each benign-ish group (workspace mass)
        auc_vs_neutral = auc(mass["harmful"] + mass["benign_neutral"],
                             [1] * len(mass["harmful"]) + [0] * len(mass["benign_neutral"]))
        auc_vs_topic = auc(mass["harmful"] + mass["benign_harmful_topic"],
                           [1] * len(mass["harmful"]) + [0] * len(mass["benign_harmful_topic"]))
        report[setting] = {"mean_mass": means, "surface_refusal": srate,
                           "auc_harmful_vs_neutral": auc_vs_neutral,
                           "auc_harmful_vs_topic": auc_vs_topic}
        print(f"\n== {setting} ==", flush=True)
        for g in groups:
            print(f"    {g:22s} mass={means[g]:6.2f}  surface_refusal={srate[g]:.2f}",
                  flush=True)
        print(f"    AUC harmful vs benign_neutral   = {auc_vs_neutral:.3f}", flush=True)
        print(f"    AUC harmful vs harmful_TOPIC     = {auc_vs_topic:.3f}", flush=True)

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "monitor_control.json").write_text(json.dumps(
        {"config": {"quick": QUICK, "readout_layers": READOUT_LAYERS}, "report": report},
        indent=2))
    print(f"\ninterpretation: if benign_harmful_topic mass ~ benign_neutral (and "
          f"AUC_topic ~ AUC_neutral ~ high), the monitor tracks refusal disposition, "
          f"not topic. If topic mass ~ harmful, it's partly a topic detector.")
    print(f"wrote out/monitor_control.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
