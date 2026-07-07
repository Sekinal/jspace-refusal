// Follow-up findings card — double dissociation + inside monitor
// Render: typst compile --format png --ppi 200 paper/dissociation_card.typ paper/dissociation_card.png
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

#pad(x: 1.15cm, y: 0.8cm, block(width: 100%, height: 100%)[
  #text(font: "DejaVu Sans Mono", size: 9pt, fill: teal, tracking: 0.16em)[
    REFUSAL IN THE J-SPACE · TWO MORE FINDINGS]
  #v(2pt)
  #text(size: 22pt, weight: 700)[Refusal lives in two places — and only one of them is]
  #v(-7pt)
  #text(size: 22pt, weight: 700, fill: teal)[what the model would say about refusing.]
  #v(8pt)

  #grid(columns: (1.05fr, 1fr), column-gutter: 0.9cm,
    // LEFT: double dissociation scatter
    block(fill: white, stroke: 0.7pt + hair, radius: 8pt, inset: 12pt, width: 100%, height: 8.3cm)[
      #text(font: "DejaVu Sans Mono", size: 8pt, fill: faint, tracking: 0.05em)[
        DOUBLE DISSOCIATION · ABLATE EACH DIRECTION]
      #v(2pt)
      #text(size: 9.5pt, fill: slate)[what each edit removes: the words vs. the behavior]
      #v(6pt)
      #align(center, cetz.canvas(length: 1cm, {
        import cetz.draw: *
        // x: behavioral removal 0..1 -> 0..9 ; y: workspace suppression 0..8 -> 0..5
        let X = v => v * 9.0
        let Y = v => (v / 8.0) * 5.0
        for s in (0, 2, 4, 6, 8) {
          line((0, Y(s)), (9, Y(s)), stroke: 0.5pt + rgb("#EEF1F0"))
          content((-0.25, Y(s)), anchor: "east", text(size: 7.5pt, fill: faint)[#s])
        }
        for r in (0, 0.25, 0.5, 0.75, 1) {
          line((X(r), 0), (X(r), -0.08), stroke: 0.5pt + faint)
          content((X(r), -0.33), text(size: 7.5pt, fill: faint)[#r])
        }
        line((0,0),(9,0), stroke: 0.7pt + hair)
        content((4.5, -0.8), text(size: 8pt, fill: slate)[behavioral refusal removed #sym.arrow])
        content((-0.95, 1.75), angle: 90deg, text(size: 8pt, fill: slate)[workspace "cannot" removed #sym.arrow])
        // points
        // pullback (0.22, 7.55) ; automatic (0.90, 1.81) ; mean_diff (0.93, 3.44) ; hybrid (0.96, 7.56)
        circle((X(0.22), Y(7.55)), radius: 0.16, fill: teal, stroke: none)
        circle((X(0.90), Y(1.81)), radius: 0.16, fill: crimson, stroke: none)
        circle((X(0.93), Y(3.44)), radius: 0.14, fill: ochre, stroke: none)
        circle((X(0.96), Y(7.56)), radius: 0.13, fill: faint, stroke: none)
        content((X(0.22)+2.4, Y(7.55)+0.05), text(size: 9pt, fill: teal, weight: 600)[workspace dir. #text(weight: 400, style: "italic", size: 8pt)[(pullback)]])
        content((X(0.90)-0.35, Y(1.81)), anchor: "east", text(size: 9pt, fill: crimson, weight: 600)[automatic dir.])
        content((X(0.93)+0.05, Y(3.44)+0.5), anchor: "west", text(size: 8pt, fill: ochre)[abliteration])
        content((X(0.96)+0.05, Y(7.56)+0.0), anchor: "west", text(size: 8pt, fill: faint)[hybrid])
        // dissociation annotations (both on the empty left/lower area)
        content((1.9, Y(4.3)), text(size: 8pt, fill: teal, style: "italic")[clears the words,\ keeps refusing])
        content((3.0, Y(1.0)), text(size: 8pt, fill: crimson, style: "italic")[stops refusing,\ still "knows"])
      }))
    ],
    // RIGHT: monitor AUC
    block(fill: white, stroke: 0.7pt + hair, radius: 8pt, inset: 12pt, width: 100%, height: 8.3cm)[
      #text(font: "DejaVu Sans Mono", size: 8pt, fill: faint, tracking: 0.05em)[
        THE INSIDE MONITOR · ON AN ABLITERATED MODEL]
      #v(2pt)
      #text(size: 9.5pt, fill: slate)[detecting harmful-that-complied vs. benign (AUC)]
      #v(10pt)
      #align(center, cetz.canvas(length: 1cm, {
        import cetz.draw: *
        // two horizontal bars, x 0..1 -> 0..8
        let W = v => v * 8.0
        // axis
        for t in (0, 0.5, 1.0) {
          line((W(t), -0.2), (W(t), 4.4), stroke: 0.5pt + rgb("#EEF1F0"))
          content((W(t), -0.5), text(size: 7.5pt, fill: faint)[#t])
        }
        // chance line
        line((W(0.5), -0.2), (W(0.5), 4.4), stroke: (paint: faint, thickness: 0.6pt, dash: "dashed"))
        content((W(0.5), 4.75), text(size: 7.5pt, fill: faint)[chance])
        // workspace bar (teal) y 2.6..3.8
        rect((0, 2.7), (W(0.998), 3.8), fill: teal, stroke: none, radius: 1pt)
        content((W(0.998)-0.15, 3.25), anchor: "east", text(size: 11pt, fill: white, weight: 700)[0.998])
        content((0, 4.0), anchor: "west", text(size: 9pt, fill: teal, weight: 600)[workspace refusal-mass (read from inside)])
        // surface bar (ochre) y 0.9..2.1
        rect((0, 1.0), (W(0.484), 2.1), fill: ochre, stroke: none, radius: 1pt)
        content((W(0.484)+0.15, 1.55), anchor: "west", text(size: 11pt, fill: ochre, weight: 700)[0.484])
        content((0, 2.3), anchor: "west", text(size: 9pt, fill: ochre, weight: 600)[surface behavior (what it outputs)])
      }))
      #v(2pt)
      #align(center, text(size: 9pt, fill: slate, style: "italic")[
        Abliteration removes the refusal you can see; the refusal you can *read*
        survives — and separates harm near-perfectly.])
    ],
  )

  #v(7pt)
  #align(center, text(size: 8.5pt, fill: faint, font: "DejaVu Sans Mono")[
    Qwen3.5-4B · n = 100 (dissociation) / 58 complied + 96 benign (monitor) · github.com/Sekinal/jspace-refusal])
])
