"""Estimation parameters and file paths for the Askot resource model.

Coordinate system
-----------------
All drillhole and block coordinates are on the LOCAL MINE GRID carried
through the legacy reports (origin: M.B.C. survey station Z-1, taken as
10,000 N / 10,000 E). The grid is metric (scale 1.000, verified against an
independently surveyed block outline) and sits ~7 deg from true north, but
it is NOT a projected CRS -- do not treat X/Y as UTM eastings/northings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

BOREHOLES_CSV = DATA_RAW / "askot_boreholes_master.csv"
ASSAYS_CSV = DATA_RAW / "askot_assay_intervals.csv"
UNDERGROUND_CSV = DATA_RAW / "askot_underground_samples_3d.csv"
SECTIONS_CSV = DATA_RAW / "askot_sections.csv"

COMPOSITES_CSV = DATA_PROCESSED / "askot_composites_3d.csv"
BLOCK_MODEL_CSV = DATA_PROCESSED / "askot_block_model.csv"
COMPARISON_CSV = DATA_PROCESSED / "askot_reserve_comparison.csv"

METALS = ("Cu_pct", "Pb_pct", "Zn_pct")

#: Bulk density conventions used by the historical estimators (t/m3).
DENSITY_T_PER_M3 = {"GSI": 3.20, "MECL": 3.50}

RANDOM_STATE = 42


@dataclass(frozen=True)
class EstimationParams:
    """Every knob that changes a reported tonnage.

    Defaults reproduce the published run. The Streamlit app varies these so
    a reviewer can see how sensitive the answer is to each choice, rather
    than being handed one number with no error bars.
    """

    #: Cut-off on total metal content (TMC = Cu% + Pb% + Zn%). MECL 1992
    #: used 2% TMC, so the default matches the benchmark estimate.
    tmc_cutoff_pct: float = 2.0

    #: Cubic block edge length (m).
    block_size_m: float = 10.0

    #: Padding added around the composite bounding box before gridding (m).
    padding_m: float = 25.0

    #: A block only counts toward tonnage if a composite sample lies within
    #: this distance of its centre. Without it the Random Forest
    #: extrapolates ore-like grades across the whole padded bounding box --
    #: including barren country rock -- and overstates tonnage by ~150x.
    #: This is the ML equivalent of classical practice, where only ground
    #: inside a geologically defensible search radius is classified.
    max_distance_to_sample_m: float = 15.0

    #: Bulk density applied to blocks above cut-off (t/m3).
    density_t_per_m3: float = 3.50

    def block_volume_m3(self) -> float:
        return self.block_size_m ** 3


@dataclass(frozen=True)
class RFParams:
    """Random Forest hyper-parameters.

    Deliberately shallow and regularised: the training set is only a few
    dozen composites, so an unconstrained forest memorises individual
    intercepts and its leave-one-borehole-out score collapses.
    """

    n_estimators: int = 300
    max_depth: int = 6
    min_samples_leaf: int = 2
    random_state: int = RANDOM_STATE
    n_jobs: int = -1

    def as_kwargs(self) -> dict:
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


#: Independent classical estimates this model is benchmarked against.
#: Sources: GSI 1974-75 progress reports; UP DGM compilation; MECL Askot
#: Phase-II final report (1992); SRK/Adi Gold NI 43-101 technical report
#: (2018). `None` means the source does not state that figure.
CLASSICAL_ESTIMATES: tuple[dict, ...] = (
    dict(name="GSI (Probable, ~1974-78)", tonnage_t=767_286.55, Cu_pct=2.32,
         Pb_pct=2.64, Zn_pct=3.95, cutoff="min 1 m thickness", density=3.20,
         method="rectangle / cross-section"),
    dict(name="DGM (Probable+Possible, ~1977-85)", tonnage_t=960_000, Cu_pct=None,
         Pb_pct=None, Zn_pct=None, cutoff="not stated", density=None,
         method="not stated"),
    dict(name="MECL 1992 (Proved+Probable)", tonnage_t=803_183, Cu_pct=2.61,
         Pb_pct=2.96, Zn_pct=4.39, cutoff="2% TMC", density=3.50,
         method="cross-section"),
    dict(name="SRK/NI 43-101 (Indicated, 2018)", tonnage_t=1_860_000, Cu_pct=2.62,
         Pb_pct=None, Zn_pct=5.80, cutoff="not stated", density=None,
         method="kriging (NI 43-101)"),
)

#: Known data-quality issues that bound how precisely any of this can be read.
DATA_CAVEATS: tuple[str, ...] = (
    "Collar RL for ASK-5 is reported with four conflicting values across the "
    "four source reports; the 1980 MECL local-grid table is used as primary.",
    "Second-phase Cu assays on the 960 m level carry roughly 30% random "
    "QA/QC error; no per-sample uncertainty weighting is applied yet.",
    "No geophysical survey data survives in any of the four reports, so the "
    "estimator is driven by assay geometry alone.",
    "Composite count is a few dozen points. Leave-one-borehole-out R2 should "
    "be read as indicative of relative skill (RF vs IDW), not as a precision "
    "claim.",
)
