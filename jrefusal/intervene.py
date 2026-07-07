# Copyright 2026. Research code (not part of jlens).
"""Residual-stream ablation via forward hooks.

:class:`AblationHook` removes each layer's residual component along a refusal
basis for the duration of a forward or ``generate`` pass. For an orthonormal
basis ``Q`` ([k, d_model]) at layer ``l`` and strength ``alpha``::

    h' = h - alpha * Qᵀ (Q h)

``alpha = 1`` fully projects the span out (standard abliteration); values in
(0, 1) partially remove it, which is the knob we tune against control-prompt
J-space distortion. Nothing is written to weights, so the edit is fully
reversible — exit the context and the model is byte-for-byte the original.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from jrefusal.refusal import RefusalDirections


class AblationHook:
    """Context manager that ablates a refusal basis from residual outputs.

    Args:
        blocks: The residual blocks (``model.layers``).
        directions: Per-layer orthonormal bases to project out.
        strength: Scalar (or per-layer dict) in ``[0, 1]``. ``1`` = full
            projection.
        layers: Restrict ablation to these layers (default: all in
            ``directions``).
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        directions: RefusalDirections,
        *,
        strength: float | dict[int, float] = 1.0,
        layers: Sequence[int] | None = None,
    ) -> None:
        self._blocks = blocks
        self._directions = directions
        active = set(directions.layers if layers is None else layers)
        self._layers = sorted(active & set(directions.layers))
        self._strength = strength
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _alpha(self, layer: int) -> float:
        if isinstance(self._strength, dict):
            return float(self._strength.get(layer, 0.0))
        return float(self._strength)

    def _make_hook(self, layer: int):
        Q = self._directions.basis[layer]                 # [k, d_model]
        alpha = self._alpha(layer)

        def hook(module, inputs, output):
            if alpha == 0.0:
                return output
            tensor = output if torch.is_tensor(output) else output[0]
            q = Q.to(device=tensor.device, dtype=tensor.dtype)
            # projection of h onto span(Q): (h @ Qᵀ) @ Q  -> [..., d_model]
            proj = (tensor @ q.t()) @ q
            edited = tensor - alpha * proj
            if torch.is_tensor(output):
                return edited
            return (edited, *output[1:])

        return hook

    def __enter__(self) -> AblationHook:
        try:
            for layer in self._layers:
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
