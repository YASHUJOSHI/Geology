# Askot Cu-Pb-Zn deposit — machine-learning resource model

[![Open the live dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://askot-resource-model.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 24](https://img.shields.io/badge/tests-24%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Part of [Yashu Joshi's geology portfolio](../README.md).

A reproducible 3-D grade and tonnage model for the **Askot polymetallic
(Cu-Pb-Zn) deposit**, Pithoragarh district, Uttarakhand, India — rebuilt from
the original assay data of four exploration campaigns spanning 1974–2018, and
benchmarked against the four classical reserve estimates those campaigns
produced.

**Headline result:** a Random Forest grade estimator that is given *nothing but
sample coordinates* — no lithology, no structure, no geological
interpretation — reproduces **1.52 Mt**, squarely inside the **0.77–1.86 Mt**
envelope of four independent expert estimates made over 44 years.

| Estimate | Tonnage | Cu % | Pb % | Zn % | Method |
|---|---|---|---|---|---|
| GSI (Probable, ~1974-78) | 0.77 Mt | 2.32 | 2.64 | 3.95 | rectangle / cross-section |
| DGM (Probable+Possible, ~1977-85) | 0.96 Mt | — | — | — | not stated |
| MECL 1992 (Proved+Probable) | 0.80 Mt | 2.61 | 2.96 | 4.39 | cross-section |
| SRK / NI 43-101 (Indicated, 2018) | 1.86 Mt | 2.62 | — | 5.80 | kriging |
| **This model (Random Forest)** | **1.52 Mt** | **2.03** | **3.02** | **5.67** | RF block model |

Dashes are figures the source report does not state; they are left blank
rather than inferred.

---

## Why this project exists

Askot has been drilled four times by four organisations, and every campaign
produced a reserve figure that disagrees with the others by up to a factor of
two. The underlying data — 47 drillhole logs, interval assays, underground
channel samples, 22 geological cross-sections — exists only as **scanned paper
reports with no text layer**. In that state the deposit cannot be re-estimated,
audited, or compared against modern practice by anyone.

This repository does three things:

1. **Consolidates** the four reports into a single machine-readable dataset,
   recording every conflict between sources rather than silently picking one.
2. **Re-estimates** the resource with a spatially cross-validated ML grade
   model, benchmarked against inverse-distance weighting and against the
   classical estimates.
3. **Publishes** the result as an interactive dashboard where every parameter
   that moves the answer — cut-off, density, search radius, block size — is a
   control rather than a buried constant.

The scientific finding worth stating plainly: **at Askot the binding
constraint is data quality, not algorithm choice.** See
[The honest finding](#the-honest-finding) below.

---

## Quick start

```bash
git clone https://github.com/YASHUJOSHI/Geology.git
cd Geology/askot-ml-resource-model
python -m venv .venv && .venv\Scripts\activate     # Windows
# source .venv/bin/activate                        # macOS / Linux
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Run the estimate from the command line:

```bash
python run_pipeline.py
python run_pipeline.py --cutoff 4.0 --block-size 5 --density 3.2
python run_pipeline.py --skip-cv --write-blocks
```

Run the tests:

```bash
pytest
```

VS Code users get three ready launch configurations (dashboard, default
pipeline, 5 m-block pipeline) from `.vscode/launch.json` — press **F5**.

---

## Method

```
scanned reports ──► consolidated CSVs ──► 3-D composites
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
          leave-one-borehole-out CV                      Random Forest fitted
          (Random Forest vs IDW)                         on all composites
                                                                      │
                                                                      ▼
                                                      grade predicted onto a
                                                      10 m cubic block grid
                                                                      │
                                                   ┌──────────────────┴────────┐
                                                   ▼                           ▼
                                        search-radius filter           TMC cut-off
                                        (≤ 15 m to a sample)           (≥ 2 % TMC)
                                                   └──────────────┬────────────┘
                                                                  ▼
                                                   tonnage = blocks × volume × density
                                                                  │
                                                                  ▼
                                                   benchmark vs GSI / DGM / MECL / SRK
```

**Grade estimator.** Multi-output Random Forest regression of (Cu %, Pb %,
Zn %) on (X, Y, Z), deliberately shallow (`max_depth=6`,
`min_samples_leaf=2`) because the training set is only 73 composites from 23
holes. An unconstrained forest memorises individual intercepts and its
held-out score collapses.

**Validation is leave-one-borehole-out, not k-fold.** Every composite from a
hole is held out together. Plain k-fold would place intervals from the same
hole in both train and test, leak position information, and report a score
that would not survive on the next drillhole. Random Forest is scored against
**inverse-distance weighting** on identical folds — the ML model has to beat
the classical interpolator to justify itself.

**The search radius is not a cosmetic parameter.** A block counts toward
tonnage only if a composite sample lies within 15 m of its centre. Without
that filter the forest extrapolates ore-like grade across the entire padded
bounding box — including barren country rock — and overstates tonnage by more
than two orders of magnitude. This is the ML equivalent of classical practice,
where only ground inside a geologically defensible search radius is classified
as Measured/Indicated/Inferred. Of 68,150 blocks in the grid, 67,715 are
discarded on this basis; 435 survive both the radius and the cut-off. The
dashboard renders the rejected volume as a faint halo so the discard is
visible rather than hidden.

**Independent check.** 40 graded intercepts digitised from 22 geological
cross-sections (Plates IV-A to IV-I of the 1992 report) give a
true-thickness-weighted grade of Cu 2.48 / Pb 4.05 / Zn 6.53 % — bracketing
the model's Cu 2.03 / Pb 3.02 / Zn 5.67 % by a method the model never sees.

---

## The honest finding

Adding 48 underground channel samples to the 25 borehole composites moved
tonnage from **1.14 Mt to 1.52 Mt** — more realistic, closer to the centre of
the classical envelope — but made cross-validated R² **worse** across all
three metals (Pb fell from −0.01 to −0.96).

That is not a bug, and it is reported rather than tuned away. Two causes:

- Channel samples resolve fine-scale (nugget) grade variation at ~2 m spacing
  that widely spaced boreholes never see, which makes held-out generalisation
  genuinely harder.
- Second-phase Cu assays on the 960 m level carry roughly **30 % random QA/QC
  error**, documented in the source report.

So more data improved the tonnage estimate and degraded the interpolation
score. The conclusion is that **no amount of algorithm selection fixes Askot —
the limit is assay quality and sample spacing.** The natural remedy,
per-sample uncertainty weighting using the documented QA/QC error rates, is
flagged as future work, not quietly implemented.

A model that only reported the flattering half of this would be less useful.

---

## Limitations

- **73 composites is a small training set.** Leave-one-borehole-out R² is
  indicative of *relative* skill (RF vs IDW); it is not a precision claim.
  Negative R² values mean neither interpolator beats the sample mean on
  held-out holes, which is an honest statement about data spacing.
- **Coordinates are on a local mine grid, not a projected CRS.** The grid is
  metric (scale 1.000, verified against an independently surveyed block
  outline) and sits ~7° from true north. A rigid 3-point fit to UTM 44N
  achieves ±36 m RMS — adequate for a location figure, not for collar-accurate
  georeferencing.
- **The block model spans only the volume bracketed by samples** plus 25 m
  padding. It is not a substitute for the classical estimates' geological
  judgement about lens continuity (Lenses I–V) between holes.
- **ASK-5 collar RL is reported with four conflicting values** across the four
  reports; the 1980 MECL local-grid table is used as primary and the conflict
  is recorded in the data, not erased.
- **No geophysical data survives** in any of the four reports, so the
  estimator is driven by assay geometry alone.

---

## Repository layout

```
askot-ml-resource-model/
├── app.py                 Streamlit dashboard
├── run_pipeline.py         CLI: fit → validate → estimate → benchmark
├── askot/
│   ├── config.py          estimation parameters, paths, benchmark figures
│   ├── data.py            loaders for composites, collars, sections
│   ├── model.py           Random Forest, IDW, leave-one-borehole-out CV
│   └── blocks.py          block grid, search radius, tonnage, grade-tonnage
├── tests/                 24 tests covering estimator invariants + the app
├── data/
│   ├── raw/               consolidated drillhole, assay, channel, section CSVs
│   └── processed/         3-D composites, benchmark table, grade-tonnage curve
└── docs/
    └── data_provenance.md how each field was reconciled across four reports
```

The block model CSV is **not** committed — it is a derived 3 MB artefact.
Regenerate it with `python run_pipeline.py --write-blocks`.

---

## Tests

24 tests, run with `pytest`. They are not smoke tests: each guards a mistake
that would silently produce a plausible-but-wrong tonnage, which is the
failure mode that matters in resource estimation because nothing crashes.

Representative invariants:

- tonnage never increases as cut-off rises; grade never decreases
- tonnage scales linearly with bulk density
- relaxing the search radius inflates tonnage by >5× (the filter is load-bearing)
- IDW predictions stay inside the convex hull of the sample values
- chunked nearest-neighbour distance matches the single-pass result
- the default run still reproduces 1.52 Mt
- the dashboard executes and its cut-off slider actually reduces tonnage

---

## Data sources

| Campaign | Organisation | Contribution |
|---|---|---|
| 1974–75 | Geological Survey of India | initial drilling, Probable reserve |
| 1980 | Mineral Exploration Corporation Ltd | local-grid collar survey, interval assays |
| 1992 | MECL, Askot Phase-II final report | 22 cross-sections, channel samples, Proved+Probable |
| 2018 | SRK / Adi Gold | NI 43-101 Indicated resource |

All four are scanned documents without a text layer; every figure in
`data/raw/` was transcribed and cross-checked against the others, with
conflicts recorded in `docs/data_provenance.md` rather than resolved silently.

---

## Author

**Yashu Joshi** · Exploration Geologist · M.Tech Geoexploration (IIT Bombay)

Ten years as an exploration geologist with Mineral Exploration and Consultancy
Ltd (MECL), a Government of India enterprise, across phosphate, gold,
base-metal, lithium and REE projects from reconnaissance (UNFC G4) through
general exploration (G3).

A companion manuscript based on this model is in preparation for submission to
*Ore Geology Reviews*.

Part of [github.com/YASHUJOSHI/Geology](https://github.com/YASHUJOSHI/Geology).

## License

MIT — see [LICENSE](LICENSE). The underlying exploration reports are the
property of their respective issuing organisations (GSI, MECL, DGM, SRK);
only derived, transcribed numerical data is included here.
