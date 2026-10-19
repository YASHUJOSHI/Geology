"""Loaders for the consolidated Askot drilling and sampling data.

The four source reports (GSI 1974-75, MECL 1980, MECL 1992, SRK 2018) are
scanned paper documents with no text layer. The CSVs read here are the
transcribed, cross-checked consolidation of those reports -- see
``docs/data_provenance.md`` for how each field was reconciled.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_composites(path=None) -> pd.DataFrame:
    """Return 3-D composite samples with usable Cu/Pb/Zn grades.

    Rows with a blank assay in any metal are dropped: they cannot serve as a
    regression target. ``borehole_id`` is retained because it is the grouping
    key for spatial cross-validation.
    """
    df = pd.read_csv(path or C.COMPOSITES_CSV)
    for metal in C.METALS:
        df[metal] = pd.to_numeric(df[metal], errors="coerce")
    for axis in ("X", "Y", "Z"):
        df[axis] = pd.to_numeric(df[axis], errors="coerce")

    df = df.dropna(subset=["X", "Y", "Z", *C.METALS]).reset_index(drop=True)
    df["TMC_pct"] = df[list(C.METALS)].sum(axis=1)
    return df


def load_collars(path=None) -> pd.DataFrame:
    """Return borehole collar positions on the local mine grid.

    ``local_grid_departure`` is the easting-equivalent and
    ``local_grid_latitude`` the northing-equivalent; both are renamed to
    X/Y so collars and blocks share one coordinate convention.
    """
    df = pd.read_csv(path or C.BOREHOLES_CSV)
    df = df.rename(columns={
        "local_grid_departure": "X",
        "local_grid_latitude": "Y",
        "collar_RL_m": "Z",
    })
    for axis in ("X", "Y", "Z"):
        df[axis] = pd.to_numeric(df[axis], errors="coerce")

    keep = ["borehole_id", "program", "X", "Y", "Z", "depth_m", "inclination_deg"]
    keep = [c for c in keep if c in df.columns]
    return df.dropna(subset=["X", "Y", "Z"])[keep].reset_index(drop=True)


def load_sections(path=None) -> pd.DataFrame:
    """Return mineralised intercepts digitised from the 1992 cross-sections.

    Used as an independent check on the block model: a true-thickness
    weighted grade from the sections should bracket the model's grade.
    """
    df = pd.read_csv(path or C.SECTIONS_CSV)
    for col in ("true_thickness_m", *C.METALS, "TMC_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def section_weighted_grade(sections: pd.DataFrame) -> dict[str, float]:
    """True-thickness weighted average grade of the graded section intercepts."""
    df = sections.dropna(subset=["true_thickness_m", *C.METALS])
    if df.empty:
        return {}
    w = df["true_thickness_m"]
    out = {m: float((df[m] * w).sum() / w.sum()) for m in C.METALS}
    out["TMC_pct"] = sum(out.values())
    out["n_intercepts"] = int(len(df))
    return out
