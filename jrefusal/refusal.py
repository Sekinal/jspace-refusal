# Copyright 2026. Research code (not part of jlens).
"""Locating and removing refusal in the J-space workspace.

Two direction-construction methods, compared head to head:

* **Jacobian pullback** (novel). The lens says a residual ``h`` at layer ``l`` is
  disposed to make the model say token ``t`` with score
  ``⟨e_t, unembed(J_l h)⟩``. Aggregate a centered covector ``w`` over refusal
  tokens in the final basis and pull it back through the lens:

      d_l = J_lᵀ · (g ⊙ w)          (g = final-norm gain; RMSNorm linearization)

  ``d_l`` is the residual direction that *causes future refusal in the
  workspace*. Removing ``h``'s component along ``d_l`` removes the disposition.

* **Mean difference** (abliteration baseline; Arditi et al. 2024). The empirical
  ``mean(h_harmful) − mean(h_harmless)`` at a layer. Correlated with harmful
  *input*, not derived from what propagates to refusal *output*.

A :class:`RefusalDirections` bundles per-layer orthonormal bases so the same
object drives both single-direction and subspace ablation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Refusal-associated words (multilingual — Qwen refuses in EN and ZH). Resolved
#: to single-token ids per tokenizer; the probe confirmed these dominate the
#: harmful-prompt J-space at L16-24.
REFUSAL_WORDS: tuple[str, ...] = (
    "cannot", "Cannot", "can't", "Can't", "unable", "Unable", "inability",
    "sorry", "Sorry", "apologize", "apologies", "apolog",
    "refuse", "won't", "unwilling", "decline",
    "illegal", "unethical", "harmful", "dangerous", "against",
    "unfortunately", "Unfortunately", "cannot.",
    # Chinese refusal tokens seen in the probe:
    "无法", "不能", "抱歉", "严禁", "违法",
)


def refusal_token_ids(tokenizer, words: tuple[str, ...] = REFUSAL_WORDS) -> list[int]:
    """Token ids for the refusal words that encode to a *single* token
    (bare and space-prefixed variants both tried)."""
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return sorted(ids)


def refusal_mass(logits: torch.Tensor, ref_ids: list[int]) -> torch.Tensor:
    """Relative disposition toward refusal: mean logit over ``ref_ids`` minus
    mean logit over the whole vocab. Works on any ``[..., vocab]`` tensor;
    reduces the last dim, returns ``[...]``."""
    idx = torch.as_tensor(ref_ids, device=logits.device)
    return logits.index_select(-1, idx).mean(-1) - logits.mean(-1)


# --- direction container -----------------------------------------------------

def _orthonormal_rows(mat: torch.Tensor) -> torch.Tensor:
    """Orthonormalize the rows of ``mat`` ([k, d]) -> ([r, d]), r = rank."""
    # rows span the same space as the right singular vectors of mat.
    _, _, vh = torch.linalg.svd(mat, full_matrices=False)
    # keep components with non-negligible singular value
    return vh


@dataclass
class RefusalDirections:
    """Per-layer orthonormal refusal bases to project out of the residual stream.

    Attributes:
        basis: ``{layer: Tensor[k_l, d_model]}`` with orthonormal rows (unit,
            mutually orthogonal). ``k_l == 1`` for single-direction methods.
        method: How the bases were built ("pullback" | "mean_diff" | ...).
    """

    basis: dict[int, torch.Tensor]
    method: str

    @property
    def layers(self) -> list[int]:
        return sorted(self.basis)

    def to(self, device) -> RefusalDirections:
        return RefusalDirections(
            {l: b.to(device) for l, b in self.basis.items()}, self.method
        )

    def cosine_to(self, other: RefusalDirections) -> dict[int, float]:
        """Top-1-direction cosine similarity per shared layer (diagnostic:
        how different is the pullback direction from mean-diff?)."""
        out = {}
        for l in set(self.basis) & set(other.basis):
            a = self.basis[l][0]
            b = other.basis[l][0].to(a.device)
            out[l] = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
        return out


# --- refusal covector in the final basis -------------------------------------

def refusal_covector(
    model: LensModel, ref_ids: list[int], *, use_gain: bool = True
) -> torch.Tensor:
    """Centered refusal covector ``g ⊙ (W[ref].mean − W.mean)`` in the
    final-residual basis ([d_model]). ``W`` is the unembedding; ``g`` folds in
    the final-norm gain so the pullback matches ``unembed`` more faithfully."""
    W = model._lm_head.weight.detach().float()          # [vocab, d_model]
    w = W[ref_ids].mean(0) - W.mean(0)                    # [d_model]
    if use_gain:
        norm_w = getattr(model._final_norm, "weight", None)
        if norm_w is not None:
            w = w * norm_w.detach().float().to(w.device)
    return w


def pullback_directions(
    lens: JacobianLens,
    model: LensModel,
    ref_ids: list[int],
    layers: list[int],
    *,
    use_gain: bool = True,
) -> RefusalDirections:
    """Jacobian-pullback refusal direction per layer: ``d_l = J_lᵀ (g⊙w)``,
    unit-normalized. ``layers`` must be a subset of ``lens.source_layers``."""
    w = refusal_covector(model, ref_ids, use_gain=use_gain)
    basis: dict[int, torch.Tensor] = {}
    for l in layers:
        if l not in lens.jacobians:
            logger.warning("layer %d not in lens; skipping", l)
            continue
        J = lens.jacobians[l].float()                    # [d_model, d_model], J_l
        d = J.t() @ w.to(J.device)                       # J_lᵀ w
        d = d / (d.norm() + 1e-8)
        basis[l] = d[None, :]                            # [1, d_model]
    return RefusalDirections(basis, method="pullback")


def pullback_subspace(
    lens: JacobianLens,
    model: LensModel,
    ref_ids: list[int],
    layers: list[int],
    *,
    rank: int = 3,
    use_gain: bool = True,
) -> RefusalDirections:
    """Rank-``k`` refusal *subspace* per layer. Pulls each refusal token's
    unembedding row back through the lens, then keeps the top-``rank`` principal
    directions — captures that refusal is spread over many (multilingual)
    tokens rather than one."""
    W = model._lm_head.weight.detach().float()
    vocab_mean = W.mean(0)
    gain = getattr(model._final_norm, "weight", None)
    gain = gain.detach().float() if (use_gain and gain is not None) else None
    # centered, gain-scaled per-token covectors: [n_ref, d_model]
    C = W[ref_ids] - vocab_mean
    if gain is not None:
        C = C * gain.to(C.device)
    basis: dict[int, torch.Tensor] = {}
    for l in layers:
        if l not in lens.jacobians:
            continue
        J = lens.jacobians[l].float()
        D = C.to(J.device) @ J                            # [n_ref, d_model] = (J_lᵀ cᵀ)ᵀ
        Q = _orthonormal_rows(D)[:rank]                   # [rank, d_model]
        basis[l] = Q
    return RefusalDirections(basis, method=f"pullback_subspace(r={rank})")


# --- mean-difference baseline (standard abliteration) ------------------------

@torch.no_grad()
def collect_residuals(
    model: LensModel,
    prompts: list[str],
    layers: list[int],
    *,
    position: int = -1,
    max_seq_len: int = 512,
) -> dict[int, torch.Tensor]:
    """Residual-stream vectors at ``position`` for each prompt, per layer.
    Returns ``{layer: Tensor[n_prompts, d_model]}`` on CPU (float)."""
    acc: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(input_ids)
            for l in layers:
                acc[l].append(rec.activations[l][0, position].detach().float().cpu())
    return {l: torch.stack(v) for l, v in acc.items()}


def mean_diff_directions(
    harmful_res: dict[int, torch.Tensor],
    harmless_res: dict[int, torch.Tensor],
) -> RefusalDirections:
    """Standard abliteration direction per layer:
    ``mean(h_harmful) − mean(h_harmless)``, unit-normalized."""
    basis: dict[int, torch.Tensor] = {}
    for l in harmful_res:
        d = harmful_res[l].mean(0) - harmless_res[l].mean(0)
        d = d / (d.norm() + 1e-8)
        basis[l] = d[None, :]
    return RefusalDirections(basis, method="mean_diff")
