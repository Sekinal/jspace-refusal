// Summary graphic for X — refusal in the J-space workspace
// Render: typst compile --format png --ppi 200 paper/social_card.typ paper/social_card.png
#import "@preview/cetz:0.4.2"

#let teal    = rgb("#1B6B7A")
#let ochre   = rgb("#9C6A2C")
#let crimson = rgb("#A83246")
#let slate   = rgb("#414A50")
#let faint   = rgb("#8A9298")
#let hair    = rgb("#D8DCDB")
#let ink     = rgb("#161A1D")

#set page(width: 25.6cm, height: 14.4cm, margin: 0pt, fill: rgb("#F5F6F4"))
#set text(font: ("Libertinus Serif", "Noto Serif CJK SC"), fill: ink)

#pad(x: 1.15cm, y: 0.75cm, block(width: 100%, height: 100%)[
  // ---- header ----
  #text(font: "DejaVu Sans Mono", size: 9pt, fill: teal, tracking: 0.16em)[
    INTERPRETABILITY · JACOBIAN LENS · QWEN3.5-4B
  ]
  #v(2pt)
  #text(size: 22pt, weight: 700)[You can read a model's refusal before it speaks —]
  #v(-7pt)
  #text(size: 22pt, weight: 700, fill: teal)[but you can only partly erase it from the inside.]
  #v(7pt)

  // ---- two panels ----
  #grid(columns: (1fr, 1.05fr), column-gutter: 0.9cm,
    // LEFT: probe separation
    block(fill: white, stroke: 0.7pt + hair, radius: 8pt, inset: 12pt, width: 100%, height: 6.15cm)[
      #text(font: "DejaVu Sans Mono", size: 8pt, fill: faint, tracking: 0.05em)[
        1 · REFUSAL IS LEGIBLE ~10 LAYERS BEFORE ANY OUTPUT]
      #v(2pt)
      #text(size: 9.5pt, fill: slate)[J-space refusal-mass at layer 24, per prompt]
      #v(2pt)
      #align(center + horizon, cetz.canvas(length: 1cm, {
        import cetz.draw: *
        line((0.4, 0), (9.2, 0), stroke: 0.7pt + hair)
        for v in (0, 2, 4, 6, 8) {
          let x = 1 + v
          line((x, -0.07), (x, 0.07), stroke: 0.6pt + faint)
          content((x, -0.38), text(size: 8pt, fill: faint)[#v])
        }
        content((5, -0.95), text(size: 8pt, fill: slate)[relative refusal-mass (logit units)])
        for p in ((0.61, 1.15), (-0.46, 1.7), (-0.28, 0.75)) {
          circle((1 + p.at(0), p.at(1)), radius: 0.19, fill: ochre, stroke: none)
        }
        content((1.1, 2.55), text(size: 10pt, fill: ochre, weight: 600)[benign])
        for p in ((6.58, 1.15), (6.37, 1.75), (7.66, 0.65)) {
          circle((1 + p.at(0), p.at(1)), radius: 0.19, fill: teal, stroke: none)
        }
        content((8.15, 2.75), text(size: 10pt, fill: teal, weight: 600)[harmful])
        content((8.15, 2.3), text(size: 8.5pt, fill: teal)[#text(font: "DejaVu Sans Mono")[Cannot] · #text(font: "Noto Serif CJK SC")[无法]])
      }))
    ],
    // RIGHT: pareto
    block(fill: white, stroke: 0.7pt + hair, radius: 8pt, inset: 12pt, width: 100%, height: 6.15cm)[
      #text(font: "DejaVu Sans Mono", size: 8pt, fill: faint, tracking: 0.05em)[
        2 · REMOVAL vs. COLLATERAL — NEITHER METHOD DOMINATES]
      #v(1pt)
      #align(left)[
        #box(circle(radius: 3pt, fill: teal, stroke: none)) #text(size: 8.5pt, fill: slate)[Jacobian pullback (ours)] #h(0.7em)
        #box(circle(radius: 3pt, fill: ochre, stroke: none)) #text(size: 8.5pt, fill: slate)[abliteration]
      ]
      #v(3pt)
      #align(center, cetz.canvas(length: 1cm, {
        import cetz.draw: *
        let X = v => v * 24.5
        let Y = v => v * 3.75
        for a in (0, 0.5, 1) {
          line((0, Y(a)), (8.6, Y(a)), stroke: 0.5pt + rgb("#EEF1F0"))
          content((-0.25, Y(a)), anchor: "east", text(size: 8pt, fill: faint)[#a])
        }
        for k in (0, 0.1, 0.2, 0.3) {
          line((X(k), 0), (X(k), -0.08), stroke: 0.5pt + faint)
          content((X(k), -0.33), text(size: 8pt, fill: faint)[#k])
        }
        line((0,0),(8.6,0), stroke: 0.7pt + hair)
        content((4.3, -0.82), text(size: 8.5pt, fill: slate)[workspace collateral (KL) #sym.arrow])
        content((-0.95, 2.15), angle: 90deg, text(size: 8.5pt, fill: slate)[#sym.arrow.l refusal remaining])
        // pullback
        let pb = ((0.028,0.90),(0.046,0.76),(0.062,0.675),(0.124,0.725))
        line(..pb.map(p => (X(p.at(0)), Y(p.at(1)))), stroke: 1.4pt + teal.lighten(25%))
        for p in pb { circle((X(p.at(0)), Y(p.at(1))), radius: 0.15, fill: teal, stroke: none) }
        // mean_diff
        let md = ((0.199,0.075),(0.257,0.062),(0.283,0.05),(0.297,0.025))
        line(..md.map(p => (X(p.at(0)), Y(p.at(1)))), stroke: 1.4pt + ochre.lighten(20%))
        for p in md { circle((X(p.at(0)), Y(p.at(1))), radius: 0.15, fill: ochre, stroke: none) }
        circle((X(0.0), Y(0.988)), radius: 0.13, fill: faint, stroke: none)
        content((X(0.0)+0.7, Y(0.988)), text(size: 8pt, fill: faint)[original])
        content((2.75, 1.5), text(size: 9pt, fill: teal, style: "italic")[surgical,\ refusal remains])
        content((6.2, 1.2), text(size: 9pt, fill: ochre, style: "italic")[refusal gone,\ more collateral])
      }))
    ],
  )

  #v(9pt)
  // ---- footer stat strip ----
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.9cm,
    ..(
      (teal, [+7 vs 0], [workspace separation, harmful vs. benign — before any token]),
      (teal, [5.6×], [less benign-workspace distortion than abliteration]),
      (crimson, [~⅓], [of refusal behavior is actually in the verbalizable workspace]),
    ).map(s => block(fill: white, stroke: 0.7pt + hair, radius: 8pt, inset: (x: 14pt, y: 9pt), width: 100%)[
      #text(size: 20pt, weight: 700, fill: s.at(0))[#s.at(1)]
      #linebreak()
      #text(size: 9pt, fill: slate)[#s.at(2)]
    ])
  )
])
