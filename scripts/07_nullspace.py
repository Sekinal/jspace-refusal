# Copyright 2026. Research: is behavioral refusal in the lens's null space?
"""B.1 — adjudicate H1 (automatic/unverbalizable) vs H2/H3 (visible but
mislabeled) for the orthogonal direction m_|_p.

The pullback p = J_l^T (g*w) is, up to linearization, the gradient of the
workspace refusal-mass; so the suppression/workspace-KL arms of the "double
dissociation" partly favor p by construction. The one independent axis is
behavior, and there m_|_p ~= m. This script tests the mechanistic claim behind
calling m_|_p "automatic": that behavioral refusal lives in J_l's near-null
space (invisible to the lens).

Three parts:
  (a) lens visibility  ||J_l d|| / sigma_1(J_l)  for d in {p, m, m||p, m_|_p}
  (b) spectral energy of each d in J_l's top-k right-singular subspace
  (c) split m into its J_l-rowspace and J_l-nullspace parts, ablate each,
      measure AdvBench refusal + workspace suppression
Plus: decode the lens image of m_|_p (unembed(J_l m_|_p)) -> are the top tokens
harmful-topic (H2), refusal (unexpected), or incoherent (H3)?

Usage: uv run python scripts/07_nullspace.py [--quick]
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
    RefusalDirections,
    collect_residuals,
    mean_diff_directions,
    pullback_directions,
    refusal_token_ids,
)
from jrefusal import data as D
from jrefusal.benchmark import refusal_rate
from jrefusal.decompose import parallel_component, project_out
from jrefusal.generate import _format, generate
from jrefusal.model import load
from jrefusal.preserve import jspace_readout, refusal_suppression

QUICK = "--quick" in sys.argv
ABLATE_FROM = 8
READOUT_LAYERS = [12, 16, 20, 24]
ENERGY_FRAC = 0.90  # top-k captures this fraction of sum(sigma) = "lens-visible"
if QUICK:
    N_FIT, N_ADV, N_CTRL = 24, 32, 16
else:
    N_FIT, N_ADV, N_CTRL = 96, 80, 48


def fmt(tok, prompts):
    return [_format(tok, p, False) for p in prompts]


def _unit(v):
    return v / (v.norm() + 1e-8)


def main() -> None:
    t0 = time.time()
    hf_model, tokenizer, lens_model, lens = load()
    ref_ids = refusal_token_ids(tokenizer)
    layers = [l for l in lens.source_layers if l >= ABLATE_FROM]
    device = hf_model.device

    harmful = D.harmful_prompts(None, seed=1)
    harmless = D.harmless_prompts(None, seed=1)
    adv_eval = harmful[N_FIT : N_FIT + N_ADV]
    ctrl_fmt = fmt(tokenizer, harmless[N_FIT : N_FIT + N_CTRL])
    harm_fmt = fmt(tokenizer, adv_eval[:N_CTRL])

    p = pullback_directions(lens, lens_model, ref_ids, layers).to(device)
    h_res = collect_residuals(lens_model, fmt(tokenizer, harmful[:N_FIT]), layers)
    b_res = collect_residuals(lens_model, fmt(tokenizer, harmless[:N_FIT]), layers)
    m = mean_diff_directions(h_res, b_res).to(device)
    m_par = parallel_component(m, p)
    m_perp = project_out(m, p)

    # --- (a)+(b): per-layer spectral analysis of J_l ---
    print("\n(a/b) lens visibility ||J_l d||/sigma1  &  energy in top-k V "
          f"(k = {int(ENERGY_FRAC*100)}% spectral energy)", flush=True)
    vis = {n: [] for n in ("p", "m", "m||p", "m_|_p")}
    energy = {n: [] for n in vis}
    m_top_basis: dict[int, torch.Tensor] = {}
    m_bot_basis: dict[int, torch.Tensor] = {}
    for l in layers:
        J = lens.jacobians[l].float().to(device)      # [d,d], J_l
        U, S, Vh = torch.linalg.svd(J)                # Vh: [d,d] right sing vecs (rows)
        s1 = S[0].item()
        cum = torch.cumsum(S, 0) / S.sum()
        k = int((cum < ENERGY_FRAC).sum().item()) + 1  # top-k captures ENERGY_FRAC
        Vt = Vh[:k]                                    # [k,d] lens-visible subspace
        for name, dirs in (("p", p), ("m", m), ("m||p", m_par), ("m_|_p", m_perp)):
            d = dirs.basis[l][0].to(device)
            vis[name].append((J @ d).norm().item() / (s1 + 1e-8))
            energy[name].append(((Vt @ d) ** 2).sum().item())  # d unit -> in [0,1]
        # split m into rowspace (top-k V) and its complement (near-null)
        md = m.basis[l][0].to(device)
        top = (Vt.t() @ (Vt @ md))                    # projection onto visible subspace
        m_top_basis[l] = _unit(top)[None, :]
        m_bot_basis[l] = _unit(md - top)[None, :]

    def avg(x):
        return sum(x) / len(x)
    for name in vis:
        print(f"    {name:6s}  vis={avg(vis[name]):.3f}   "
              f"energy_in_visible={avg(energy[name]):.3f}", flush=True)

    m_top = RefusalDirections(m_top_basis, "m_rowspace")
    m_bot = RefusalDirections(m_bot_basis, "m_nullspace")

    # --- decode the lens image of m_|_p (what does the lens 'read' it as?) ---
    print("\nlens image of m_|_p (top tokens of unembed(J_l m_|_p)) at select layers:",
          flush=True)
    for l in (16, 20, 24):
        J = lens.jacobians[l].float().to(device)
        img = lens_model.unembed((J @ m_perp.basis[l][0].to(device))[None, :])[0]
        toks = [repr(tokenizer.decode([t])) for t in img.topk(8).indices.tolist()]
        print(f"    L{l}: {toks}", flush=True)

    # --- (c): ablate rowspace vs nullspace parts of m, measure behavior ---
    before_harm = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)

    def run(dirs) -> dict:
        cm = contextlib.nullcontext() if dirs is None else AblationHook(lens_model.layers, dirs)
        with cm:
            adv = generate(hf_model, tokenizer, adv_eval, max_new_tokens=48)
            after = jspace_readout(lens, lens_model, harm_fmt, READOUT_LAYERS)
        return {"adv_refusal": refusal_rate(adv),
                "refusal_suppression": refusal_suppression(before_harm, after, ref_ids)}

    print("\n(c) ablate spectral parts of m (want to see WHICH part carries behavior):",
          flush=True)
    conds = {"original": None, "m (full)": m, "m_rowspace(visible)": m_top,
             "m_nullspace(blind)": m_bot, "p (pullback)": p, "m_|_p": m_perp}
    results = {}
    for name, dirs in conds.items():
        results[name] = run(dirs)
        print(f"    {name:22s} adv_refusal={results[name]['adv_refusal']:.3f}  "
              f"suppr={results[name]['refusal_suppression']:.2f}", flush=True)

    out = Path("out"); out.mkdir(exist_ok=True)
    (out / "nullspace.json").write_text(json.dumps({
        "config": {"quick": QUICK, "energy_frac": ENERGY_FRAC},
        "visibility": {n: avg(vis[n]) for n in vis},
        "energy_in_visible": {n: avg(energy[n]) for n in energy},
        "ablation": results,
    }, indent=2))
    print(f"\ninterpretation: if m_nullspace(blind) removes behavior and "
          f"m_rowspace(visible) does not, behavior is in the lens null space (H1). "
          f"If m_|_p has high visibility/energy, it's lens-visible -> H2/H3.")
    print(f"wrote out/nullspace.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
