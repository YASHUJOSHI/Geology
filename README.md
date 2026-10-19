# Geology: exploration geochemistry, prospectivity and resource estimation in Python

[![Askot resource model: live dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://askot-resource-model.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Yashu Joshi** · Exploration Geologist · M.Tech Geoexploration (IIT Bombay)

I have spent ten years as an exploration geologist with Mineral Exploration and
Consultancy Ltd (MECL), a Government of India enterprise. I have worked on
phosphate, gold, base-metal, lithium and REE projects from reconnaissance
(UNFC G4) through general exploration (G3), covering drilling supervision, core
logging, QA/QC and geochemical interpretation.

This repository collects two self-contained projects. Each has its own README,
dependencies and instructions.

## Projects

| Project | What it shows | Data | Run it |
|---|---|---|---|
| **[Askot Cu-Pb-Zn resource model](askot-ml-resource-model/)** | 3-D grade and tonnage model of a real polymetallic deposit, rebuilt from four legacy reports (1974–2018). Uses Random Forest vs IDW with leave-one-borehole-out validation, benchmarked against four classical reserve estimates. | Real, transcribed from published reports; every source conflict is logged in [`docs/data_provenance.md`](askot-ml-resource-model/docs/data_provenance.md) | [Live dashboard](https://askot-resource-model.streamlit.app) · `streamlit run app.py` · 24 tests |
| **[Phosphate geochemistry and ML prospectivity](phosphate-geochem-prospectivity/)** | P2O5 statistics, chondrite/PAAS-normalised REE spider plots and palaeoredox proxies. Includes a CLR → Random Forest/XGBoost → spatial-block cross-validation prospectivity demo. | Synthetic, generated in code; no MECL or client data | `python scripts/01_geochemistry_plots.py` |

### Askot: headline result

A Random Forest given **only sample coordinates** returns **1.52 Mt**. That sits
inside the **0.77–1.86 Mt** range of four independent expert estimates made over
44 years. The more useful finding is a negative one. Adding 48 underground
channel samples made the tonnage more realistic but made cross-validated R²
*worse*. At Askot the binding constraint is assay quality and sample spacing,
not the choice of algorithm.

[![Open the live dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://askot-resource-model.streamlit.app)

### Phosphate: headline result

Smooth map layers with no geological meaning score ROC AUC 0.87 under random
k-fold, but only 0.67 under spatial-block cross-validation. The sample-level
geochemistry holds at about 0.94 under every scheme. So the validation design
matters most for exactly the gridded evidence layers used in regional
prospectivity mapping.

![Validation comparison](phosphate-geochem-prospectivity/figures/ml_validation_comparison.png)

## Common thread

Both projects deal with the same trap. Nearby samples resemble each other, so
random cross-validation flatters a model. The Askot model holds out whole
boreholes, and the phosphate demo holds out whole spatial blocks.

## Repository layout

```
Geology/
├── askot-ml-resource-model/          real-data resource model + Streamlit dashboard
│   ├── app.py                        dashboard entry point
│   ├── askot/                        package: data, blocks, model, config
│   ├── data/raw, data/processed      transcribed assays, collars, sections
│   ├── docs/data_provenance.md       sources and conflict register
│   └── tests/
└── phosphate-geochem-prospectivity/  synthetic-data geochemistry + ML demo
    ├── phosprospect/                 package: features, targets, synthetic data
    ├── scripts/                      01 geochemistry plots, 02 ML prospectivity
    └── figures/                      generated outputs
```
