"""phosprospect - geochemical targeting for sedimentary phosphorite deposits."""

from .features import build_features, clr_transform
from .targets import (
    authigenic_score, grade_class, indicator_detail,
    label_from_lithology, flag_log_chemistry_conflict,
)
from .reference import PAAS_REE_PPM, PAAS_TRACE_PPM

__version__ = "0.1.0"
__all__ = [
    "build_features", "clr_transform", "authigenic_score", "grade_class",
    "indicator_detail", "label_from_lithology", "flag_log_chemistry_conflict",
    "PAAS_REE_PPM", "PAAS_TRACE_PPM",
]
