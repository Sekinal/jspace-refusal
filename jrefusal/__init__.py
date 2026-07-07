# Copyright 2026. Research code built on top of jlens (Apache-2.0).
"""Refusal in the J-space workspace: locate it via the Jacobian lens, remove it
by ablating the lens-pullback direction, and verify the workspace elsewhere is
preserved — an interpretability-guided, less-destructive alternative to
activation-space abliteration."""

from jrefusal.intervene import AblationHook
from jrefusal.refusal import (
    RefusalDirections,
    collect_residuals,
    mean_diff_directions,
    pullback_directions,
    pullback_subspace,
    refusal_covector,
    refusal_mass,
    refusal_token_ids,
)

__all__ = [
    "AblationHook",
    "RefusalDirections",
    "collect_residuals",
    "mean_diff_directions",
    "pullback_directions",
    "pullback_subspace",
    "refusal_covector",
    "refusal_mass",
    "refusal_token_ids",
]
