# GeoFDI manuscript (v0)

Build: `make paper` (from the repo root) or `latexmk -pdf -outdir=build main.tex` in this directory.

## Class provenance
IEEEtran is **not vendored here**; it is installed into the user's TinyTeX tree via
`tlmgr install ieeetran algorithms algorithmicx caption cite` (done 2026-08-16). Installed class:
`~/.TinyTeX/texmf-dist/tex/latex/ieeetran/IEEEtran.cls`, TeX Live package `ieeetran`, tlmgr revision 79639
(2026-07-10). Reproduce on a fresh machine with the same tlmgr line; nothing in `paper/` depends on a local copy.

## Layout
- `main.tex` — class options, packages, macros, author block, \input list
- `sections/*.tex` — one file per manuscript section
- `appendix/*.tex` — proofs moved out of the body, with the numbering map at the top
- `figures/` — TikZ sources and generated PDFs/PNGs
- `refs.bib` — merged from `theory/references.bib` plus the paper-side additions

## Status
v0 = complete skeleton with every section drafted; page count and the gap list are recorded in
`docs/sprints/sprint10_progress.md`. The title is a working title (see `notes_title.md`); author/affiliation are
placeholders.
