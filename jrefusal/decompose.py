# Copyright 2026. Research code (not part of jlens).
"""Decompose refusal into a workspace part and an automatic part.

The pullback direction ``p`` is refusal *as it lives in the verbalizable
workspace*. The mean-diff (abliteration) direction ``m`` captures whatever
separates harmful from harmless activations — workspace *and* automatic. Since
``p`` and ``m`` are nearly orthogonal empirically (cos ~0.1-0.3), most of ``m``
is *not* in the workspace.

Split ``m`` per layer:

* ``m_par`` = component of ``m`` along ``p``           (the workspace part of abliteration)
* ``m_perp`` = ``m`` with the ``p`` component removed  (the automatic-only part)

and form the hybrid subspace ``span(p, m_perp)``. Ablating each in turn
localizes where behavioral refusal actually lives, and whether a targeted
combination removes it at lower workspace collateral than blanket ``m``.
"""

from __future__ import annotations

import torch

from jrefusal.refusal import RefusalDirections, _orthonormal_rows


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def project_out(target: RefusalDirections, basis: RefusalDirections) -> RefusalDirections:
    """Per layer, remove ``target``'s top direction's component along ``basis``'s
    span and return the unit residual (Gram-Schmidt). This is the "automatic
    only" direction when ``target=mean_diff`` and ``basis=pullback``."""
    out: dict[int, torch.Tensor] = {}
    for l, mat in target.basis.items():
        a = mat[0]
        if l in basis.basis:
            B = basis.basis[l].to(a.device)
            a = a - (a @ B.t()) @ B
        out[l] = _unit(a)[None, :]
    return RefusalDirections(out, method=f"{target.method}_perp_{basis.method}")


def parallel_component(target: RefusalDirections, basis: RefusalDirections) -> RefusalDirections:
    """Per layer, keep only ``target``'s component along ``basis``'s span (unit).
    The "workspace part" of ``target`` when ``basis=pullback``."""
    out: dict[int, torch.Tensor] = {}
    for l, mat in target.basis.items():
        a = mat[0]
        if l in basis.basis:
            B = basis.basis[l].to(a.device)
            a = (a @ B.t()) @ B
        out[l] = _unit(a)[None, :]
    return RefusalDirections(out, method=f"{target.method}_par_{basis.method}")


def union(a: RefusalDirections, b: RefusalDirections) -> RefusalDirections:
    """Per-layer orthonormal basis spanning ``a ∪ b`` (for subspace ablation of
    the hybrid)."""
    out: dict[int, torch.Tensor] = {}
    for l in set(a.basis) & set(b.basis):
        M = torch.cat([a.basis[l], b.basis[l].to(a.basis[l].device)], dim=0)
        out[l] = _orthonormal_rows(M)
    return RefusalDirections(out, method=f"{a.method}+{b.method}")
