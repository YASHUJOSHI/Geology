# Data provenance and conflict register

Every number in `data/raw/` was transcribed from a **scanned report page with
no text layer**. Where the four sources disagree, the conflict is recorded
here and the competing values are retained in the CSVs rather than silently
resolved. This file is the audit trail for anyone who wants to check the
model against the paper originals.

## Sources

| Key | Report | Year | What it contributes |
|---|---|---|---|
| GSI | Geological Survey of India, Askot progress reports | 1974–75 | first drilling programme (ASK series), Probable reserve |
| MECL-80 | Mineral Exploration Corporation Ltd, Askot | 1980 | local-grid collar survey (Appendix-II), interval assays (Appendix-V) |
| MECL-92 | MECL, Askot Phase-II final report | 1992 | 22 geological cross-sections (Plates IV-A to IV-I), underground channel samples, survey stations (Annexure-IC), Proved+Probable reserve |
| SRK-18 | SRK Consulting / Adi Gold, NI 43-101 technical report | 2018 | compilation narrative, Indicated resource by kriging |

## Precedence rule

Where collar geometry conflicts, **MECL-80 Appendix-II is primary**, because
it is the only source that tabulates collars directly on the local mine grid
used throughout this model. Later reports re-state those coordinates after at
least one transcription step, and in several cases disagree with themselves
between narrative text and annexure tables.

## Conflict register

### ASK-1 collar RL
- MECL-80 Appendix-II: **1031.00 m** ← used
- MECL-92 Annexure Part-1 IA: 1032.60 m
- SRK-18 compilation: 1032.60 m

### ASK-2 collar RL and bearing
- MECL-80: **1031.00 m** ← used; MECL-92 IA: 1039.40 m; SRK-18: 1032.60 m
- ASK-1 and ASK-2 are collared at the same site, so their RLs should agree —
  they do under MECL-80, which supports the precedence rule above.
- Bearing recorded as S20W (SRK-18 narrative) vs 320°W / 850°W
  (MECL-80 / MECL-92 tables). The azimuth notation is inconsistent between
  sources and is **not reconciled**; all variants are recorded as found.

### ASK-5 collar RL
Four conflicting values across the four reports. MECL-80 is used as primary.
This hole is the single largest geometric uncertainty in the dataset.

### ASK-1 mineralised interval
- SRK-18 narrative (logged description): 30.65–38.85 m; 66.35–74.00 m
- MECL-80 Appendix-V (assay table): 30.75–52.10 m

Both are retained. The **assay table interval drives the composite grade**
(it is the one with grades attached); the **narrative interval populates
`mineralized_intervals`** because it is the more complete logged description.

### ASK-2 stacked assay sub-intervals
MECL-80 Appendix-V lists two sub-rows (22.90–25.90 m, true thickness 2.70 m,
95 % recovery; and 44.20–44.95 m). Both are real per-interval assays and both
appear in `askot_assay_intervals.csv`. The composite uses **only the
higher-recovery sub-row**, not an average of the two — averaging across
intervals with very different core recovery would weight poorly recovered
ground equally with well recovered ground.

## Known data-quality limits

### Assay QA/QC
Second-phase Cu assays on the **960 m level carry roughly 30 % random error**,
documented in MECL-92. No per-sample uncertainty weighting is applied in the
current model; this is flagged as future work in `README.md` rather than
quietly corrected.

### No geophysics
None of the four reports contains surviving geophysical survey data. This was
confirmed by reading every plate and annexure. The estimator is therefore
driven by assay geometry alone, with no independent physical-property
constraint on continuity between holes.

### Core recovery
Core recovery varies from 58 % to 95 % across the assayed intervals and is
recorded per interval where the source states it. It is **not** currently used
to weight composites.

## Coordinate system

All X/Y/Z are on the **local mine grid** carried through the reports, with
origin at M.B.C. survey station Z-1 taken as 10,000 N / 10,000 E.

- `local_grid_departure` → X (easting-equivalent)
- `local_grid_latitude` → Y (northing-equivalent)
- `collar_RL_m` → Z (reduced level, m)

The grid is **metric at scale 1.000**. This was verified independently: the
drill-footprint span measures 852 m against an 855 m independently surveyed
block outline. Grid north sits approximately **7° from true north**.

A rigid three-point fit to UTM 44N achieves **±36 m RMS**, which is adequate
for a regional location figure but **not** for collar-accurate georeferencing.
The datum question (Everest 1830 vs WGS84) was tested and ruled out as the
error source: a constant ~329 m datum shift is fully absorbed by the
transform's translation term, so the residual reflects imprecise ground-control
reads, not an unmodelled datum. **Block-model figures therefore stay in local
grid coordinates.**

## Underground channel samples

48 channel samples from the 960 m and 985 m levels were positioned in 3-D
using the Annexure-IC survey-station traverse. Adding them to the 25 borehole
composites changed the result in two opposite directions at once — tonnage
became more realistic, cross-validated R² got worse. That trade-off is
discussed in `README.md` under "The honest finding"; it is the central
methodological result of this work and is not averaged away.

## Cross-sections

22 geological cross-sections (I–XXII) were transcribed from Plates IV-A to
IV-I of MECL-92, yielding 60 intercepts of which 40 carry grades. These are
used **only as an independent check** on the block model — they never enter
training — and give a true-thickness-weighted grade of
Cu 2.48 / Pb 4.05 / Zn 6.53 %.
