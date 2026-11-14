# Experimental status

The old `main.tex` and chapter files describe the retired building-type
classification pilot. Their claims (53 categories, class-balanced focal loss,
test-as-validation results and component-boundary interpretation) must not be
submitted or cited as semantic-segmentation evidence. The new standalone
`S7_MANUSCRIPT_DRAFT.tex` is an internal review draft for the 31-part study.

The active task is the 31-label point-wise segmentation pipeline documented
in `../RESEARCH_WORKFLOW.md`. The frozen S7 backbone has completed its
confirmatory and authorized test evaluation. S8 failed its three-seed internal
promotion gate; it has no test result. The new draft attributes S7's measured
gain to its tested global block under a fixed density-mass backbone, not to
density mass independently. It needs scientific review and literature work
before submission; the old `main.tex` remains retired.

## CAMO-Build Prototype 1 status (2026-09-21)

Prototype 1 (measure weighting + Local Measure-Aware Operator, no OT/router/
hierarchy) has completed 6/6 full runs on hardware. Density-mode mass
weighting improved val mIoU by +0.00071 (+0.56%) on 3/3 seeds, but this is
**directional evidence only** — not statistically significant (paired t-test
p = 0.0673). See `../PROTOTYPE1_EXECUTION_REPORT.md` for full numbers.

The 2026-09-21 Prototype 2 plan was historical and is superseded by the
completed S7/S8 sequence above. Do not reuse the phrase "local conservative
operator" for the Local Measure-Aware Operator: its row-normalized aggregation
does not conserve the mass-weighted feature integral.
