# Sprint 10 progress (gate checklist)

Read this first in a new session; continue from the first unchecked item. `[ ]` open / `[x] <commit>` done /
`[~] <commit>` done with a documented miss. Update and push at the end of every Block. Spec: `sprint10_spec.md`.

## Anti-loss
- [ ] spec + progress committed

## Block M0 — metadata finalisation + three closing experiments
- [ ] M0.1 site labels confirmed by operator (xb=A, nmb=B, by=C); all [inferred] flipped to [confirmed]; R2 table reissued
- [ ] M0.2 pre-registration for the xb4/nmb3 split experiment committed BEFORE the run
- [ ] M0.2 split experiment: xb4 by RTK indoor/outdoor half, nmb3 by first/second half; 2x2 table + verdict
- [ ] M0.3 B1 friction row: analytic-residual R⁻ on friction_scale x{1.5,2,3}, R=20, merged into the baseline table
- [ ] M0.4 progress registers: raw/m1/nominal-crossday/ pending user download; two emails pending
- [ ] rp034 Block M0 review pack

## Block W — full manuscript v0 (tag `paper-draft-v0`)
- [ ] W1 paper/ scaffolding: IEEEtran (source + version recorded), main.tex + sections/ + appendix/, merged bib, `make paper`
- [ ] W2.1 title (+ two alternates in paper/notes_title.md)
- [ ] W2.2 Fig. 1 concept figure (TikZ, single column)
- [ ] W2.3 Algorithm 1 + nine correctness conditions (each tagged with a theorem/experiment id)
- [ ] W2.4 Introduction full draft (problem / gap / thesis / four contributions / numeric result preview)
- [ ] W2.5 Claims table (Table I): 6 detectors x 8 properties, every cell tagged with its experiment id
- [ ] W3 Preliminaries / Theory / Method / Simulation / External / Hardware / Limitations / Conclusion
- [ ] W3 hardware injection placeholder box matches the robot-day list one-to-one
- [ ] W gate: `make paper` builds a complete PDF, zero errors; page count + gap list recorded
- [ ] rp035 Block W review pack; tag `paper-draft-v0`

## Block N — onboard proxy benchmark
- [ ] N1 `scripts/bench_pipeline.py` (per-stage timing, single core, BLAS=1, ms/cycle + sustainable Hz vs 250 Hz)
- [ ] N2 laptop run on one Go2 + one M1 session; two tables
- [ ] N3 `docs/protocol/bench_nuc.md` + `env/machines/nuc13.yaml` ready for tomorrow's NUC run
- [ ] rp036 Block N review pack

## Wrap-up
- [ ] final summary: page count + TOC, placeholder-box list, bench table, still-open decisions
