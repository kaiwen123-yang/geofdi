# Sprint 10 progress (gate checklist)

Read this first in a new session; continue from the first unchecked item. `[ ]` open / `[x] <commit>` done /
`[~] <commit>` done with a documented miss. Update and push at the end of every Block. Spec: `sprint10_spec.md`.

## Anti-loss
- [x] 0adfad0 spec + progress committed

## Block M0 — metadata finalisation + three closing experiments
- [x] M0.1 labels CONFIRMED exactly as inferred; propagated to meta form/audit/catalog/11 meta.yaml/e20 config; **R2 numbers unchanged** (annotation-only reissue, reported)
- [x] 84379c1 M0.2 pre-registration committed before the run (states the unverifiable indoor/outdoor labelling up front)
- [x] M0.2 split: **neither pre-registered outcome — localised non-stationarity** (one half clean, one alarms, in both). xb4 symmetric readouts identical (1 %/3 %) ⇒ surface-switch NOT supported; nmb3 foot force +17 %. Rule sharpened to per-HOMOGENEOUS-segment
- [x] M0.3 friction row (x1.2/1.5/2/3): raw R⁻ 0.00 → **residual R⁻ 1.00 at x1.5, FAR 0.00** vs classical 0.85 at FAR 0.50; merged into baseline_protocol.md
- [x] M0.4 registered: `raw/m1/nominal-crossday/` still awaiting the user's download; two emails pending (user's action)
- [x] rp034 Block M0 review pack

## Block W — full manuscript v0 (tag `paper-draft-v0`)
- [x] W1 paper/ scaffolding; IEEEtran via tlmgr with provenance in paper/README.md; 5 paper-side refs each verified against publisher/arXiv; `make paper`
- [x] W2.1 title + two alternates + the open naming decision
- [x] W2.2 Fig. 1 TikZ concept figure (single column)
- [x] W2.3 Algorithm 1 (8 lines) + Table II nine conditions, each tagged; C9 (signed statistic for side) is the new one
- [x] W2.4 Introduction full draft with measured current-state evidence and the numeric preview
- [x] W2.5 Table I claims matrix, every cell tagged, with an explicit 'not a dominance claim' paragraph
- [x] W3 all sections drafted + appendix proofs with the numbering map
- [x] W3 framed placeholder box P1–P7 mirrors the robot-day list one-to-one
- [x] W gate: **10 pages, 0 errors, 0 undefined refs**; sections I–X + appendix
- [x] rp035 Block W review pack; tag `paper-draft-v0`

## Block N — onboard proxy benchmark
- [x] N1 `scripts/bench_pipeline.py` (BLAS pinned before numpy import; no extra deps)
- [x] N2 laptop: Go2 3.38 ms/cycle = **131× real time**, InEKF 6137 Hz vs 249 Hz; M1 2.25 ms/block = **444×**
- [x] N3 bench_nuc.md (three steps + laptop reference table) + env/machines/nuc13.yaml
- [x] rp036 Block N review pack

## Wrap-up
- [x] final summary delivered
