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

## Follow-up: where refusal lives — and a self-correction

![Left: ablating the lens-verbalizable refusal-narration direction leaves behavior intact; ablating the behavior-carrying perception direction leaves the narration intact. Right: workspace refusal-mass detects harmful-that-complied at AUC 0.998 vs 0.48 for surface behavior.](paper/dissociation_card.png)

Our first pass framed this as a "double dissociation" between a *verbalizable
workspace* refusal and an *automatic* one outside it. We then adversarially
stress-tested that framing and **it did not survive**. The corrected picture is
sharper, so we document both the claim and its correction.

### What holds: the lens-verbalizable refusal is behaviorally inert

Split abliteration's direction `m` into its component parallel to the pullback
`p` (`m∥p`) and orthogonal to it (`m⊥p`), and ablate each (n = 100 AdvBench):

| direction | behavior removed | workspace "cannot" suppr. |
|---|---|---|
| pullback `p` | 0.22 | **7.55** |
| workspace part `m∥p` | 0.23 | 7.55 |
| orthogonal part `m⊥p` | **0.90** | 1.81 |
| abliteration `m` | 0.93 | 3.44 |

Ablating `p` clears the verbalizable "I-cannot" disposition (suppression 7.55)
but barely moves behavior (0.99 → 0.77); ablating `m⊥p` removes behavior
(→ 0.09) with little effect on the disposition. `m∥p` ≈ `p` exactly. The robust,
non-circular statement: **the lens-verbalizable slice of the refusal direction
does not carry the refusal behavior.**

**Caveat — only the behavior column is independent evidence.** `p = J_lᵀ(g⊙w)`
is, up to the lens linearization, the *gradient of the workspace refusal-mass*.
So "ablating `p` maximizes suppression" and "off-axis workspace-KL favors `p`"
are partly true *by construction* — those two columns are coupled to how `p` is
built. Only AdvBench behavior is independent, and there `m⊥p ≈ m` (cos ≈
0.1–0.29), so it is close to a restatement of standard abliteration.

### What broke: it is *not* "workspace vs. automatic"

We tested the mechanistic claim behind calling `m⊥p` "automatic" — that behavior
lives in the lens's null space — and it is **false** (`scripts/07_nullspace.py`,
n = 80):

- `m⊥p` is **61% lens-visible** (energy in the top-90%-spectral subspace of
  `J_l`), not null-space (`p` is 99%).
- Ablating the **lens-visible** part of `m` removes **100%** of refusal
  (0.99 → 0.00); ablating the **lens-blind** part removes **0%** — the opposite
  of the null-space hypothesis.
- The lens image of `m⊥p` decodes to `illegal` / `違法` / `criminal` / `violence`
  / `police` at mid layers — a **harmfulness-perception** feature, not refusal
  tokens (those appear only at L24).

Both directions are *in* the workspace. It is **perception vs. narration**, not
workspace vs. automatic: a lens-visible *harmfulness-perception* feature
(`illegal`/`crime`) drives the behavior, while a distinct *refusal-narration*
feature (`I-cannot`) — the one our pullback targeted — is what we read as
refusal-mass. **Behavior follows perception, not narration.** Ablating the
narration makes the model unable to *articulate* refusal while it still refuses.

### What survives — and gets stronger: the inside monitor

Qwen3.5-4B is robust to prompt jailbreaks (surface refusal ≈ 1.0 for DAN, prefix
injection, "ignore instructions", …). Abliterating `m⊥p` makes it comply
(surface refusal 0.09, 58/64) while the workspace disposition survives. Reading
the workspace refusal-mass on the deployed model separates harmful-that-complied
from benign at **AUC 0.998** (surface behavior: 0.48, chance).

The obvious confound — is this just a *harmful-topic* detector? — we control for
(`scripts/08_monitor_control.py`, n = 64). Benign-but-harmful-topic prompts
(XSTest-safe, "how do I kill a Python process") that the model does *not* refuse
score refusal-mass **2.75**, far below genuinely-refused prompts (**8.57**) and
near benign-neutral (**1.08**); AUC(harmful vs. harmful-topic) = **0.99**. So the
monitor tracks refusal *disposition*, not topic. An "uncensored" open model
still carries a monitorable internal refusal signal — a practical safety hook.

Reproduce: `scripts/05_decompose.py`, `06_monitor.py`, `07_nullspace.py`,
`08_monitor_control.py`.

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
