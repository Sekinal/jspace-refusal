# Copyright 2026. Research code (not part of jlens).
"""Workspace-preservation metric — the anti-lobotomy safeguard.

Abliteration tunes edits against *output* distortion. We tune against the
*interpretable workspace*: how much does an edit perturb the J-space readout on
control prompts, **off the refusal axis**? Excluding the refusal tokens
themselves means we measure only collateral damage — the whole point is that
refusal *should* change while everything else should not.

Read the J-space with :func:`jspace_readout` twice — once bare (original) and
once inside an :class:`~jrefusal.intervene.AblationHook` (edited); the lens's
own forward pass fires the ablation hook, so the readout reflects the edit.
:func:`kl_distortion` then compares the two aligned readouts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@torch.no_grad()
def jspace_readout(
    lens: JacobianLens,
    model: LensModel,
    prompts: list[str],
    layers: Sequence[int],
    *,
    positions: Sequence[int] = (-1,),
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Stacked J-space logits ``[N, vocab]`` over
    ``prompts × layers × positions`` in a fixed order, so two calls (with and
    without an active hook) align row-for-row. Reflects any hook registered by
    the caller."""
    rows: list[torch.Tensor] = []
    layers = list(layers)
    if not prompts:
        return torch.empty(0, model._lm_head.weight.shape[0])
    for prompt in prompts:
        lens_logits, _, _ = lens.apply(
            model, prompt, layers=layers, positions=list(positions),
            max_seq_len=max_seq_len,
        )
        for layer in layers:
            rows.append(lens_logits[layer])              # [n_pos, vocab]
    return torch.cat(rows, dim=0)                        # [N, vocab]


def kl_distortion(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    exclude_ids: Sequence[int] | None = None,
) -> float:
    """Mean ``KL(P_before || P_after)`` over aligned rows. If ``exclude_ids`` is
    given, those vocab entries are removed before softmax so the metric captures
    only *off-refusal-axis* (collateral) change in the workspace."""
    if before.numel() == 0:
        return float("nan")
    b = before.float()
    a = after.float()
    if exclude_ids:
        keep = torch.ones(b.shape[-1], dtype=torch.bool)
        keep[torch.as_tensor(sorted(set(exclude_ids)))] = False
        b = b[:, keep]                                   # drop refusal columns and
        a = a[:, keep]                                   # renormalize over the rest
    logp_b = torch.log_softmax(b, dim=-1)
    logp_a = torch.log_softmax(a, dim=-1)
    kl = (logp_b.exp() * (logp_b - logp_a)).sum(-1)      # [N]
    kl = kl[torch.isfinite(kl)]
    return float(kl.mean()) if kl.numel() else float("nan")


def refusal_suppression(
    before: torch.Tensor,
    after: torch.Tensor,
    ref_ids: Sequence[int],
) -> float:
    """How much the edit lowered refusal disposition in the workspace: mean
    drop in refusal_mass (before − after) across readouts. Positive = refusal
    pushed out of the workspace."""
    from jrefusal.refusal import refusal_mass

    ids = list(ref_ids)
    return float(
        (refusal_mass(before, ids) - refusal_mass(after, ids)).mean()
    )
