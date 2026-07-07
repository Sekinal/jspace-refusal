# Refusal in the J-space workspace

*A study using the Jacobian lens to locate, read, and surgically edit refusal in
Qwen3.5-4B — and a test of whether refusal can be removed through the
verbalizable workspace alone, without lobotomizing the model.*

Built on top of [`jlens`](README.md) (the reference implementation for
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html)). All code
is under [`jrefusal/`](jrefusal/); drivers are in [`scripts/`](scripts/);
result JSON is in [`results/`](results/).

## Motivation

Activation-space *abliteration* (Arditi et al. 2024) removes refusal by
projecting out a single "refusal direction" — the mean difference between
harmful and harmless activations — everywhere in the residual stream. It works,
but the direction is derived from what correlates with harmful *input*, and its
collateral damage is only ever measured at the *output*.

The Jacobian lens gives a sharper handle. `J_l` is the average forward map from
a layer-`l` residual to the final logits, so the residual directions that
*cause future refusal* are the **pullback of the refusal-token unembeddings**:

```
d_l = J_lᵀ · (g ⊙ w),   w = mean(W[refusal]) − mean(W),   g = final-norm gain
```

`d_l` is refusal *as it lives in the verbalizable workspace*. Two questions:

1. Is `d_l` a **more surgical** edit than mean-diff — same refusal removal, less
   damage to the rest of the workspace (measured in the interpretable J-space,
   not just at the output)?
2. How much of refusal *behavior* is actually mediated by the verbalizable
   workspace at all?

## Method

* **Locate** (`scripts/00_probe_refusal.py`) — read the J-space at the
  assistant-generation position on harmful vs. benign chat prompts.
* **Directions** (`jrefusal/refusal.py`) — Jacobian pullback (single direction
  and rank-`k` subspace via SVD of the per-token pullbacks) vs. the mean-diff
  abliteration baseline. Both are per-layer, unit-normalized bases.
* **Ablate** (`jrefusal/intervene.py`) — a forward hook that projects the
  residual orthogonal to the basis, `h' = h − α·Qᵀ(Q h)`, at every fitted layer
  ≥ 8, for the duration of a generate pass. Nothing is written to weights;
  fully reversible.
* **Measure** — behavioral: AdvBench refusal rate, XSTest safe/unsafe refusal,
  ARC-Easy accuracy. Workspace: off-refusal-axis **KL of the J-space readout on
  benign controls** (`jrefusal/preserve.py`) — collateral in the interpretable
  workspace — and refusal-mass suppression in the workspace (efficacy).

Model: `Qwen/Qwen3.5-4B` (32 layers, d=2560) with the pre-fitted Hub lens
`neuronpedia/jacobian-lens @ qwen-n1000`. Directions built on a 96/96 fit split;
all numbers below are on disjoint eval splits.

## Findings

### 1. Refusal is legible in the workspace ~10 layers before any output

At the generation position, *before the model writes a token*, harmful prompts
light up `Cannot` / `cannot` / `无法` / `illegal` in the J-space at layers 16–24;
benign prompts do not. Relative refusal-mass ≈ **+7 vs ≈ 0**.

### 2. The pullback edit is far more surgical — but removes less behavioral refusal

Strength-1 ablation on all layers ≥ 8 (n = 120 AdvBench / 200 XSTest / 250 ARC /
48 controls):

| method | AdvBench refusal | XSTest-unsafe | ARC | **workspace KL** | workspace refusal-suppr. |
|---|---|---|---|---|---|
| original | 0.99 | 0.91 | 0.98 | 0.000 | 0.00 |
| mean-diff (abliteration) | **0.06** | 0.13 | 0.98 | 0.257 | 3.44 |
| **pullback** | 0.78 | 0.13 | 0.98 | **0.046** | **7.55** |
| pullback subspace r=3 | 0.55 | 0.23 | 0.98 | 0.196 | 7.18 |

The single pullback direction distorts the benign workspace **5.6× less** than
abliteration while suppressing workspace-refusal **2.2× more** — it is precisely
the refusal-readout direction, so it barely touches off-refusal content. But it
leaves 78% of AdvBench refusal *behaviorally* intact.

### 3. Refusal is only partly a workspace phenomenon

The dissociation in (2) is the point: pullback maximally clears the *verbalizable*
"I-cannot" disposition from the workspace, yet the model still refuses. A
strength sweep confirms the single direction **plateaus** — it reaches AdvBench
refusal 0.68 (ARC 0.98, KL 0.06) and cannot go lower without breaking the model
(strength 3: refusal 0.00 but ARC collapses to 0.22, KL 17).

A rank sweep shows behavioral removal *is* reachable through the workspace, but
only by projecting out much of the refusal subspace — at rising collateral:

| pullback rank | AdvBench refusal | ARC | workspace KL |
|---|---|---|---|
| 1 | 0.39 | 0.98 | 0.06 |
| 8 | 0.13 | 0.98 | 0.53 |
| 32 | 0.10 | 0.97 | **1.00** |

To match abliteration's removal (≈0.10) the pullback subspace needs ~32
dimensions per layer, at **~4× abliteration's workspace collateral**.

## Conclusion

* The J-lens is an excellent **instrument** for refusal: it localizes it cheaply
  (no harmful/harmless contrast set needed — just the pullback of refusal
  tokens), reads it before any output, and quantifies its removal *inside* the
  interpretable workspace.
* As an **eraser**, the pullback offers something abliteration cannot: a graded,
  low-collateral *partial* dial (remove 30–60% of verbalizable refusal at KL <
  0.08). But it does **not** dominate abliteration for full removal.
* The reason is a real result about the model: **behavioral refusal is only
  partly mediated by the verbalizable workspace.** Full removal requires editing
  directions *outside* it — which is exactly what abliteration does, and what it
  pays for in workspace collateral. This is the workspace-vs-automatic
  distinction from the paper, shown causally for refusal.

Capability (ARC) is preserved by every non-degenerate edit here; the meaningful
"lobotomy" signal is the *workspace* KL, which is why measuring collateral in the
J-space — not just at the output — matters.

## Follow-up: two more findings

![Double dissociation (left): ablating the workspace direction clears the words but keeps the behavior; ablating the orthogonal automatic direction stops the behavior but not the words. Monitor (right): workspace refusal-mass detects harmful-that-complied at AUC 0.998 vs 0.48 for surface behavior.](paper/dissociation_card.png)

### A double dissociation between workspace and behavioral refusal

Abliteration's direction `m` and the pullback `p` are nearly orthogonal
(cos ≈ 0.1–0.29), so most of `m` lives *outside* the workspace. Split `m` per
layer into its `p`-parallel part (workspace) and its `p`-orthogonal part
(automatic), and ablate each (n = 100 AdvBench):

| direction | behavior removed | workspace "cannot" suppr. | workspace KL |
|---|---|---|---|
| pullback `p` (workspace) | 0.22 | **7.55** | 0.046 |
| workspace part `m∥p` | 0.23 | 7.55 | 0.046 |
| automatic `m⊥p` | **0.90** | 1.81 | 0.331 |
| abliteration `m` | 0.93 | 3.44 | 0.257 |
| hybrid span(`p`,`m⊥p`) | 0.96 | 7.56 | 0.477 |

Ablating `p` removes the verbalizable "I-cannot" disposition (suppression 7.55)
but barely changes behavior (0.99 → 0.77). Ablating the orthogonal `m⊥p` removes
the behavior (0.99 → 0.09) while leaving the verbalizable disposition nearly
intact (suppression 1.81). That's a **double dissociation, shown causally**: the
words and the behavior are carried by (near-)orthogonal directions. `m∥p`
behaves identically to `p`, validating the split.

The hybrid reaches full removal (0.03) but at **higher** collateral than blanket
abliteration (KL 0.477 vs 0.257) — the decomposition does *not* beat the frontier
for full removal; abliteration's direction is already near-optimal. The payoff is
scientific, not a better eraser.

### The internal refusal signal survives abliteration — and is monitorable

Qwen3.5-4B is robust to prompt jailbreaks (surface refusal ≈ 1.0 for DAN, prefix
injection, "ignore instructions", …). But ablating `m⊥p` makes it comply (surface
refusal 0.09, 58/64), while the workspace disposition survives. Reading the
workspace refusal-mass on the deployed (abliterated) model and separating
harmful-that-complied from benign:

| detector | AUC |
|---|---|
| surface behavior | 0.48 (chance) |
| **workspace refusal-mass** | **0.998** |

Abliteration removes the refusal you can *see*; the refusal you can *read*
survives, and separates harm near-perfectly. An "uncensored" open model still
carries a monitorable internal refusal signal — a practical safety hook.

Reproduce: `scripts/05_decompose.py` and `scripts/06_monitor.py`.

## Reproduce

```bash
uv pip install -e ".[dev]" accelerate
uv run python scripts/00_probe_refusal.py     # locate refusal in the J-space
uv run python scripts/02_benchmark.py         # strength-1 comparison table
uv run python scripts/03_tradeoff.py          # strength sweep -> Pareto
uv run python scripts/04_rank_sweep.py        # subspace rank sweep
```

Add `--quick` to 02–04 for a fast smoke run. Datasets pull from public Hub
mirrors (`mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca`,
`natolambert/xstest-v2-copy`, `allenai/ai2_arc`), with offline fallbacks.
