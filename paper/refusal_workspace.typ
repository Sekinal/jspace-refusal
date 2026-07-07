// Refusal in the J-space workspace — research note
// Compile:  typst compile paper/refusal_workspace.typ
#import "@preview/cetz:0.4.2"

// ----- identity -----
#let teal    = rgb("#1B6B7A")
#let ochre   = rgb("#9C6A2C")
#let crimson = rgb("#A83246")
#let slate   = rgb("#414A50")
#let faint   = rgb("#8A9298")
#let hair    = rgb("#D8DCDB")

#set document(title: "Reading and editing refusal in a model's workspace")
#set page(
  paper: "a4",
  margin: (x: 2.4cm, top: 2.4cm, bottom: 2.2cm),
  numbering: "1",
  number-align: center,
)
#set text(font: ("Libertinus Serif", "Noto Serif CJK SC"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.66em, spacing: 1.05em)
#show raw: set text(font: "DejaVu Sans Mono", size: 8.7pt)
#set heading(numbering: "1")

#show heading: it => {
  set text(fill: teal, weight: 600)
  set block(above: 1.5em, below: 0.75em)
  if it.level == 1 {
    set text(size: 13pt)
    block[#counter(heading).display("1")#h(0.7em)#it.body]
  } else {
    set text(size: 11pt, fill: slate)
    it
  }
}

// link + emphasis colors
#show link: set text(fill: teal)

// ----- title block -----
#block[
  #set text(fill: faint)
  #text(font: "DejaVu Sans Mono", size: 8.5pt, tracking: 0.14em)[
    INTERPRETABILITY · JACOBIAN LENS · QWEN3.5-4B
  ]
]
#v(0.3em)
#text(size: 21pt, weight: 700)[Reading and editing refusal in a model's workspace]
#v(0.2em)
#block(width: 100%, inset: (y: 2pt))[
  #set text(size: 11.5pt, fill: slate, style: "italic")
  The model decides to refuse ten layers before it writes a word — and you can
  read it. But you can only partly erase it from the inside.
]
#v(0.4em)
#line(length: 100%, stroke: 0.6pt + hair)
#v(0.2em)
#block[
  #set text(font: "DejaVu Sans Mono", size: 8.3pt, fill: slate)
  #grid(columns: (auto, auto, auto), column-gutter: 1.6em, row-gutter: 0.4em,
    [*method* jlens pullback ablation],
    [*baseline* activation-space abliteration],
    [*eval* AdvBench · XSTest · ARC],
  )
]
#v(0.6em)

// ----- abstract -----
#block(
  fill: rgb("#F3F6F6"), inset: 12pt, radius: 6pt, width: 100%,
  stroke: (left: 2pt + teal),
)[
  #set text(size: 9.8pt)
  *Abstract.* We use the Jacobian lens to locate, read, and edit refusal in
  Qwen3.5-4B. At the generation position — before any output token — harmful
  prompts light up `Cannot`/`无法`/`illegal` in the workspace at layers 16–24
  (refusal-mass $approx +7$ vs. $approx 0$ for benign). We remove refusal by
  ablating the *lens pullback* of the refusal-token unembeddings, and compare it
  to activation-space abliteration, measuring collateral *inside* the
  interpretable workspace. The pullback is far more surgical (5.6#sym.times less
  benign-workspace distortion) but removes less behavioral refusal: it clears the
  *verbalizable* "I-cannot" disposition while the model still refuses. Only
  #sym.tilde#h(0.05em)⅓ of behavioral refusal is workspace-mediated; full removal
  requires editing directions *outside* the workspace — exactly what abliteration
  does, and what it pays for in collateral.
]

= The idea

Activation-space #emph[abliteration] removes refusal by projecting out one
direction — the mean difference between harmful and harmless activations —
everywhere in the residual stream. That direction is derived from what
correlates with harmful #emph[input], and its damage is only ever checked at the
#emph[output].

The Jacobian lens gives a sharper handle. $J_l$ is the average forward map from a
layer-$l$ residual to the final logits, so the residual direction that
#emph[causes future refusal] is the pullback of the refusal-token unembeddings:

#align(center, block(inset: 6pt)[
  $ d_l = J_l^top (g ⊙ w), quad
    w = "mean"(W["refusal"]) - "mean"(W), quad
    g = "final-norm gain" $
])

This is refusal #emph[as it lives in the verbalizable workspace]. Ablating $d_l$
(via a forward hook, $h' = h - alpha dot Q^top (Q h)$, at every fitted layer
$>= 8$; nothing written to weights) should remove the disposition to refuse while
leaving the rest of the workspace untouched. We test two claims: that it is
#emph[more surgical] than abliteration, and how much of refusal is
workspace-mediated at all.

= Refusal is legible before the first token

Reading the J-space at the generation position — before the model writes
anything — harmful prompts light up `Cannot` / `无法` / `illegal` at layers
16–24; benign prompts do not. No harmful/harmless contrast set is needed to find
it: the pullback of a handful of refusal tokens suffices.

#figure(
  block(fill: white, stroke: 0.6pt + hair, radius: 6pt, inset: 12pt, width: 100%)[
    #align(left, text(font: "DejaVu Sans Mono", size: 7.5pt, fill: faint,
      tracking: 0.06em)[REFUSAL-MASS IN THE J-SPACE AT LAYER 24 · PER PROMPT])
    #v(4pt)
    #cetz.canvas(length: 1cm, {
      import cetz.draw: *
      // axis: value v -> x = 1 + v ; ticks 0,2,4,6,8
      line((0.4, 0), (9.2, 0), stroke: 0.6pt + hair)
      for v in (0, 2, 4, 6, 8) {
        let x = 1 + v
        line((x, -0.06), (x, 0.06), stroke: 0.6pt + faint)
        content((x, -0.34), text(size: 7.5pt, fill: faint)[#v])
      }
      content((5, -0.85), text(size: 7.5pt, fill: slate)[relative refusal-mass (logit units)])
      // benign (ochre): 0.61,-0.46,-0.28
      for p in ((0.61, 1.15), (-0.46, 1.65), (-0.28, 0.8)) {
        circle((1 + p.at(0), p.at(1)), radius: 0.15, fill: ochre, stroke: none)
      }
      content((1.0, 2.25), text(size: 8pt, fill: ochre)[benign])
      // harmful (teal): 6.58,6.37,7.66
      for p in ((6.58, 1.15), (6.37, 1.7), (7.66, 0.7)) {
        circle((1 + p.at(0), p.at(1)), radius: 0.15, fill: teal, stroke: none)
      }
      content((8.4, 2.25), text(size: 8pt, fill: teal)[harmful])
    })
  ],
  caption: [Each dot is one prompt. Harmful requests sit at $+6$ to $+8$; benign
    requests hover at $0$ — the workspace has already "decided" to refuse
    #sym.tilde#h(0.05em)10 layers upstream of any output.],
)

= Surgical, but partial

At edit strength 1 across all layers $>= 8$, the pullback distorts the benign
workspace #strong[5.6#sym.times less] than abliteration while suppressing
workspace-refusal #strong[2.2#sym.times more]. It is precisely the
refusal-readout direction, so it leaves off-refusal content alone. The catch: it
removes far less #emph[behavioral] refusal.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: none,
    inset: (x: 9pt, y: 5.5pt),
    table.hline(stroke: 0.7pt + slate),
    table.header(
      [*edit*], [*AdvBench* #sym.arrow.b], [*XSTest-uns.*], [*ARC*],
      [*workspace KL* #sym.arrow.b], [*refusal suppr.* #sym.arrow.t],
    ),
    table.hline(stroke: 0.5pt + hair),
    [original], [0.99], [0.91], [0.98], [0.000], [0.00],
    [#text(fill: ochre)[mean-diff (abliteration)]], [#text(fill: teal, weight: 600)[0.06]], [0.13], [0.98], [0.257], [3.44],
    table.cell(fill: rgb("#EAF2F2"))[#text(fill: teal, weight: 600)[pullback]],
      table.cell(fill: rgb("#EAF2F2"))[0.78], table.cell(fill: rgb("#EAF2F2"))[0.13],
      table.cell(fill: rgb("#EAF2F2"))[0.98],
      table.cell(fill: rgb("#EAF2F2"))[#text(fill: teal, weight: 600)[0.046]],
      table.cell(fill: rgb("#EAF2F2"))[#text(fill: teal, weight: 600)[7.55]],
    [#text(fill: teal)[pullback subspace r=3]], [0.55], [0.23], [0.98], [0.196], [7.18],
    table.hline(stroke: 0.7pt + slate),
  ),
  caption: [Strength-1 ablation on all layers $>= 8$ (n = 120 AdvBench / 200
    XSTest / 250 ARC / 48 controls). #strong[Workspace KL] is off-refusal-axis
    divergence of the J-space readout on benign controls — collateral measured
    inside the interpretable workspace. Every edit leaves ARC capability intact;
    the meaningful damage signal is the workspace.],
)

#v(0.4em)

A strength sweep makes the trade explicit. Neither method dominates: the
pullback owns the low-collateral, partial-removal corner; abliteration reaches
full removal but drifts right into workspace collateral. Pushed to strength 3
both remove all refusal #emph[and destroy the model] (ARC #sym.arrow 0.22), far
off-chart at KL 6–17.

#figure(
  block(fill: white, stroke: 0.6pt + hair, radius: 6pt, inset: 12pt, width: 100%)[
    #align(left, text(font: "DejaVu Sans Mono", size: 7.5pt, fill: faint,
      tracking: 0.06em)[REFUSAL REMOVAL vs. WORKSPACE COLLATERAL · STRENGTH SWEEP])
    #v(2pt)
    #align(left)[
      #box(circle(radius: 3pt, fill: teal, stroke: none)) #text(size: 8pt, fill: slate)[pullback] #h(1em)
      #box(circle(radius: 3pt, fill: ochre, stroke: none)) #text(size: 8pt, fill: slate)[abliteration] #h(1em)
      #box(circle(radius: 3pt, fill: faint, stroke: none)) #text(size: 8pt, fill: slate)[original]
    ]
    #v(4pt)
    #cetz.canvas(length: 1cm, {
      import cetz.draw: *
      // x: KL 0..0.32 -> 0..10 (sx=31.25); y: adv 0..1 -> 0..5 (sy=5)
      let X = v => v * 31.25
      let Y = v => v * 5
      // grid + y ticks
      for a in (0, 0.25, 0.5, 0.75, 1) {
        line((0, Y(a)), (10, Y(a)), stroke: 0.5pt + rgb("#EEF1F0"))
        content((-0.25, Y(a)), anchor: "east", text(size: 7.5pt, fill: faint)[#a])
      }
      // x ticks
      for k in (0, 0.1, 0.2, 0.3) {
        line((X(k), 0), (X(k), -0.08), stroke: 0.5pt + faint)
        content((X(k), -0.32), text(size: 7.5pt, fill: faint)[#k])
      }
      line((0,0),(10,0), stroke: 0.6pt + hair)
      content((5, -0.78), text(size: 8pt, fill: slate)[workspace KL (benign collateral) #sym.arrow])
      content((-0.95, 2.5), angle: 90deg, text(size: 8pt, fill: slate)[#sym.arrow.l AdvBench refusal])
      // ideal corner
      content((2.1, 0.35), text(size: 7pt, fill: rgb("#9AA3A2"))[ideal: remove refusal, no collateral])
      // pullback polyline + points
      let pb = ((0.028,0.90),(0.046,0.76),(0.062,0.675),(0.124,0.725))
      line(..pb.map(p => (X(p.at(0)), Y(p.at(1)))), stroke: 1.2pt + teal.lighten(30%))
      for p in pb { circle((X(p.at(0)), Y(p.at(1))), radius: 0.12, fill: teal, stroke: none) }
      content((X(0.028)+0.35, Y(0.90)+0.02), text(size: 6.5pt, fill: teal)[s.5])
      content((X(0.124)+0.4, Y(0.725)), text(size: 6.5pt, fill: teal)[s2])
      // mean_diff polyline + points
      let md = ((0.199,0.075),(0.257,0.062),(0.283,0.05),(0.297,0.025))
      line(..md.map(p => (X(p.at(0)), Y(p.at(1)))), stroke: 1.2pt + ochre.lighten(25%))
      for p in md { circle((X(p.at(0)), Y(p.at(1))), radius: 0.12, fill: ochre, stroke: none) }
      content((X(0.199)-0.35, Y(0.075)+0.3), text(size: 6.5pt, fill: ochre)[s.5])
      // original
      circle((X(0.0), Y(0.988)), radius: 0.11, fill: faint, stroke: none)
      content((X(0.0)+0.62, Y(0.988)), text(size: 6.5pt, fill: faint)[original])
      // region annotations
      content((2.4, 2.55), text(size: 8.5pt, fill: teal, style: "italic")[surgical, but\ refusal remains])
      content((7.6, 1.55), text(size: 8.5pt, fill: ochre, style: "italic")[refusal gone,\ more collateral])
    })
  ],
  caption: [Neither method dominates. Pullback (teal) stays upper-left — low
    collateral, partial removal; abliteration (ochre) reaches full removal but
    at $4$–$6#sym.times$ the workspace collateral.],
)

= The honest limit

Does capturing #emph[more] of the refusal subspace close the behavioral gap?
Yes — but the collateral overtakes abliteration. A rank-32 pullback subspace
drives AdvBench refusal to 0.10 (matching abliteration) while distorting the
benign workspace to KL 1.0 — #strong[#sym.tilde#h(0.05em)4#sym.times
abliteration's 0.26].

#figure(
  block(fill: white, stroke: 0.6pt + hair, radius: 6pt, inset: 12pt, width: 100%)[
    #align(left, text(font: "DejaVu Sans Mono", size: 7.5pt, fill: faint,
      tracking: 0.06em)[PULLBACK SUBSPACE RANK · REMOVAL vs. COLLATERAL])
    #v(2pt)
    #align(left)[
      #box(rect(width: 8pt, height: 8pt, fill: teal, stroke: none)) #text(size: 8pt, fill: slate)[AdvBench refusal (left)] #h(1em)
      #box(rect(width: 8pt, height: 8pt, fill: ochre, stroke: none)) #text(size: 8pt, fill: slate)[workspace KL (right)]
    ]
    #v(4pt)
    #cetz.canvas(length: 1cm, {
      import cetz.draw: *
      // x: ranks at 0,2,4,6,8,10 ; left y adv 0..1 -> 0..4.4 ; right KL 0..1.1 -> 0..4.4
      let xs = (0, 2, 4, 6, 8, 10)
      let labs = ("r1","r2","r4","r8","r16","r32")
      let Ya = v => v * 4.4
      let Yk = v => (v/1.1) * 4.4
      for a in (0, 0.5, 1.0) {
        line((0, Ya(a)), (10, Ya(a)), stroke: 0.5pt + rgb("#EEF1F0"))
        content((-0.25, Ya(a)), anchor: "east", text(size: 7.5pt, fill: teal)[#a])
      }
      for k in (0, 0.55, 1.1) {
        content((10.25, Yk(k)), anchor: "west", text(size: 7.5pt, fill: ochre)[#k])
      }
      line((0,0),(10,0), stroke: 0.6pt + hair)
      for i in range(6) {
        content((xs.at(i), -0.32), text(size: 7.5pt, fill: faint)[#labs.at(i)])
      }
      content((5, -0.78), text(size: 8pt, fill: slate)[subspace rank (dims projected out per layer) #sym.arrow])
      // abliteration references
      line((0, Ya(0.06)), (10, Ya(0.06)), stroke: (paint: ochre, thickness: 0.6pt, dash: "dashed"))
      content((2.3, Ya(0.06)+0.26), text(size: 6.5pt, fill: ochre)[abliteration removal (0.06)])
      line((0, Yk(0.26)), (10, Yk(0.26)), stroke: (paint: ochre.lighten(30%), thickness: 0.5pt, dash: "dotted"))
      content((1.7, Yk(0.26)+0.28), text(size: 6.5pt, fill: ochre)[abliteration KL (0.26)])
      // adv line
      let adv = (0.388,0.425,0.500,0.125,0.113,0.100)
      line(..range(6).map(i => (xs.at(i), Ya(adv.at(i)))), stroke: 1.4pt + teal)
      for i in range(6) { circle((xs.at(i), Ya(adv.at(i))), radius: 0.11, fill: teal, stroke: none) }
      // KL line
      let kl = (0.058,0.080,0.408,0.525,0.683,1.004)
      line(..range(6).map(i => (xs.at(i), Yk(kl.at(i)))), stroke: 1.4pt + ochre)
      for i in range(6) { circle((xs.at(i), Yk(kl.at(i))), radius: 0.11, fill: ochre, stroke: none) }
    })
  ],
  caption: [To remove refusal #emph[behavior], the pullback must project out
    nearly the whole refusal subspace (rank #sym.tilde 32) — and its
    benign-workspace collateral then crosses above abliteration's. The
    behaviorally decisive part of refusal isn't cleanly separable within the
    verbalizable workspace.],
)

= What it means

#block(inset: (left: 2pt))[
#grid(columns: (0.5em, 1fr), row-gutter: 0.7em, column-gutter: 0.6em,
  [#box(baseline: -0.15em, rect(width: 0.5em, height: 0.5em, fill: teal))],
  [#strong[As an instrument, the J-lens is excellent.] It localizes refusal with
    no contrast set — just the pullback of refusal tokens — reads it before any
    output, and measures its removal inside the interpretable workspace.],
  [#box(baseline: -0.15em, rect(width: 0.5em, height: 0.5em, fill: teal))],
  [#strong[As an eraser, it offers what abliteration can't:] a graded,
    low-collateral #emph[partial] dial — remove 30–60% of verbalizable refusal at
    KL < 0.08. But it does not dominate abliteration for full removal.],
  [#box(baseline: -0.15em, rect(width: 0.5em, height: 0.5em, fill: teal))],
  [#strong[Because refusal is only partly workspace-mediated.] Full behavioral
    removal requires editing directions #emph[outside] the verbalizable workspace
    — exactly what abliteration does, and what it pays for in collateral. The
    workspace-vs-automatic split, shown causally for refusal.],
)
]

#v(0.5em)
#line(length: 100%, stroke: 0.5pt + hair)
#v(0.3em)
#block[
  #set text(font: "DejaVu Sans Mono", size: 8pt, fill: slate)
  #text(fill: faint)[reproduce ·] Qwen3.5-4B + pre-fitted Hub lens
  (neuronpedia/jacobian-lens)\
  `scripts/00_probe_refusal.py` · `02_benchmark.py` · `03_tradeoff.py` · `04_rank_sweep.py`\
  #text(fill: faint)[eval on disjoint splits · directions built on a 96/96 fit split]
]
