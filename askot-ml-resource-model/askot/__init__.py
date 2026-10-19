"""Machine-learning resource estimation for the Askot Cu-Pb-Zn deposit.

Reconstructs a 3-D grade and tonnage model for the Askot polymetallic
deposit (Pithoragarh district, Uttarakhand, India) from four legacy
exploration campaigns spanning 1974-2018, and benchmarks it against the
classical estimates those campaigns produced.
"""
from .config import (
    CLASSICAL_ESTIMATES,
    DATA_CAVEATS,
    METALS,
    EstimationParams,
    RFParams,
)
from .blocks import (
    ResourceEstimate,
    benchmark_table,
    build_block_model,
    build_grid,
    estimate_resource,
    grade_tonnage_curve,
    in_range_mask,
    nearest_sample_distance,
)
from .data import (
    load_collars,
    load_composites,
    load_sections,
    section_weighted_grade,
)
from .model import (
    CVResult,
    feature_importance,
    fit_random_forest,
    idw_predict,
    leave_one_borehole_out_cv,
)

__version__ = "1.0.0"

__all__ = [
    "CLASSICAL_ESTIMATES",
    "CVResult",
    "DATA_CAVEATS",
    "EstimationParams",
    "METALS",
    "RFParams",
    "ResourceEstimate",
    "benchmark_table",
    "build_block_model",
    "build_grid",
    "estimate_resource",
    "feature_importance",
    "fit_random_forest",
    "grade_tonnage_curve",
    "idw_predict",
    "in_range_mask",
    "leave_one_borehole_out_cv",
    "load_collars",
    "load_composites",
    "load_sections",
    "nearest_sample_distance",
    "section_weighted_grade",
]
