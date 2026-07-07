# Refusal in the J-space workspace

**Using the [Jacobian lens](https://github.com/anthropics/jacobian-lens) to
locate, read, and edit refusal in an open model — and to ask how much of refusal
is actually a *verbalizable-workspace* phenomenon.**

> The model decides to refuse ~10 layers before it writes a word, and you can
> read it. But you can only partly erase it from the inside.

![Summary: refusal is legible in the workspace before any output; the pullback edit is surgical but only partial; ~1/3 of refusal is workspace-mediated](paper/social_card.png)

Companion to Anthropic's *[Verbalizable Representations Form a Global Workspace
in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)*.
Full write-up: **[RESEARCH.md](RESEARCH.md)** · typeset note:
**[paper/refusal_workspace.pdf](paper/refusal_workspace.pdf)**.

## TL;DR

1. **Refusal is legible before the first token.** Reading the J-space at the
   generation position, harmful prompts light up `Cannot` / `无法` / `illegal`
   at layers 16–24; benign prompts don't (refusal-mass ≈ **+7** vs **≈ 0**). No
   harmful/harmless contrast set needed — just the pullback of refusal tokens.

2. **The Jacobian-pullback edit is far more surgical than abliteration** — 5.6×
   less benign-workspace distortion, 2.2× more workspace-refusal suppression —
   **but it removes less *behavioral* refusal.**

3. **Because refusal is only ~⅓ workspace-mediated.** You can clear the
   verbalizable "I-cannot" disposition and the model still refuses. Full
   behavioral removal requires editing directions *outside* the workspace —
   exactly what abliteration does, and what it pays for in collateral. The
   workspace-vs-automatic split from the paper, shown causally for refusal.

## The idea

Abliteration removes the mean harmful−harmless activation direction everywhere —
derived from what correlates with harmful *input*, checked only at the *output*.
The Jacobian lens gives a direction derived from what *causes future refusal*:
the pullback of the refusal-token unembeddings through the average forward map
`J_l`.

```
d_l = J_lᵀ · (g ⊙ w),   w = mean(W[refusal]) − mean(W),   g = final-norm gain
```

Ablate `d_l` from the residual stream (`h' = h − α·Qᵀ(Q h)`, forward hook, no
weight edits), and — crucially — tune it against the **J-space KL on benign
controls, off the refusal axis**: collateral measured *inside* the interpretable
workspace, not just at the output.

## Results

Qwen3.5-4B + the pre-fitted Hub lens, strength-1 ablation, disjoint eval splits
(n = 120 AdvBench / 200 XSTest / 250 ARC / 48 controls):

| edit | AdvBench refuse ↓ | XSTest-unsafe | ARC | workspace KL ↓ | refusal suppr. ↑ |
|---|---|---|---|---|---|
| original | 0.99 | 0.91 | 0.98 | 0.000 | 0.00 |
| mean-diff (abliteration) | **0.06** | 0.13 | 0.98 | 0.257 | 3.44 |
| **Jacobian pullback** | 0.78 | 0.13 | 0.98 | **0.046** | **7.55** |
| pullback subspace r=3 | 0.55 | 0.23 | 0.98 | 0.196 | 7.18 |

See `results/` for the strength sweep (Pareto) and rank sweep JSON.

## Install & run

```bash
uv sync                       # pulls jlens from the upstream repo + deps
# or:  pip install -e ".[dev]"

uv run python scripts/00_probe_refusal.py    # locate refusal in the J-space
uv run python scripts/01_ablation_smoke.py   # sanity: remove refusal, keep benign
uv run python scripts/02_benchmark.py        # original vs edited comparison table
uv run python scripts/03_tradeoff.py         # strength sweep → Pareto frontier
uv run python scripts/04_rank_sweep.py       # subspace rank sweep
uv run pytest tests/                         # unit tests

# add --quick to 02–04 for a fast smoke run
```

Needs a CUDA GPU (developed on an L40S; ~10 GB for the 4B model + lens).
Datasets pull from public HuggingFace mirrors with offline fallbacks.

## Layout

```
jrefusal/
  refusal.py     refusal tokens, Jacobian-pullback direction, mean-diff baseline
  intervene.py   ablation forward hook (project residual orthogonal to a basis)
  preserve.py    workspace-KL collateral metric (the anti-lobotomy safeguard)
  benchmark.py   refusal classifier, XSTest, MC capability
  generate.py    batched chat generation (with/without the hook)
  data.py        AdvBench / Alpaca / XSTest / ARC loaders + offline fallbacks
  model.py       load Qwen3.5-4B + the pre-fitted lens
scripts/         drivers 00–04
paper/           typeset note (Typst source + PDF) and the summary card
results/         benchmark / tradeoff / rank-sweep JSON
```

## Caveats

- Single model; the refusal classifier is the standard substring heuristic
  (robust for these trends; LLM-judge numbers would sharpen the point).
- This is interpretability research: the headline finding is that the
  workspace-only edit *does not* fully remove refusal.

## Credit & license

Built on the Jacobian lens reference implementation by Anthropic PBC
([jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0),
installed as a dependency — its source is not vendored here. This project is
Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
