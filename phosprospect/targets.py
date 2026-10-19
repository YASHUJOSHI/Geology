"""Two-stage target definition: authigenic phosphate presence, then economic grade.

Stage 1 asks whether a phosphatic system is present at all. Stage 2 asks whether it
reaches grade. Separating them matters because a diluted authigenic sample and a barren
one are geologically different but can carry identical P2O5.
"""

import numpy as np
import pandas as pd

# Terms a core log or field sheet may use for phosphatic material.
PHOSPHORITE_TERMS = (
    "phosphorite", "phosphatic", "francolite", "collophane",
    "apatite", "phosphate", "rock phosphate",
)

# Independent lines of evidence for authigenic francolite. Each entry is
# (feature, comparison, threshold). These are geological judgements, not tuned
# parameters - edit them and the meaning of the whole model changes.
AUTHIGENIC_INDICATORS = {
    "Y_Ho": (">", 33.0),
    "MREE_bulge": (">", 1.2),
    "Ce_anomaly": ("<", 0.90),
    "P_Al": (">", 1.0),
    "EF_Sr": (">", 2.0),
    "EF_U": (">", 3.0),
    "U_authigenic": (">", 0.0),
}

# Grade bands follow Indian reporting practice for rock phosphate.
DEFAULT_GRADE_BANDS = [
    (0.0, 5.0, "barren"),
    (5.0, 15.0, "sub_grade"),
    (15.0, 25.0, "low_grade"),
    (25.0, 100.0, "high_grade"),
]


def label_from_lithology(df, lithology_col="lithology", terms=PHOSPHORITE_TERMS):
    """Stage-1 label taken from the logged lithology, independent of geochemistry."""
    if lithology_col not in df.columns:
        raise KeyError(
            f"'{lithology_col}' not found. Stage-1 labels must come from logging or "
            "petrography, not from the geochemical features used as model inputs."
        )
    text = df[lithology_col].astype(str).str.lower()
    hit = pd.Series(False, index=df.index)
    for term in terms:
        hit |= text.str.contains(term, na=False, regex=False)
    return hit.astype(int).rename("authigenic")


def authigenic_score(features, indicators=None, as_fraction=True):
    """Count how many authigenic indicators a sample satisfies, ignoring missing ones."""
    ind = indicators or AUTHIGENIC_INDICATORS
    met = pd.DataFrame(index=features.index)
    available = pd.DataFrame(index=features.index)

    for name, (op, threshold) in ind.items():
        if name not in features.columns:
            continue
        col = pd.to_numeric(features[name], errors="coerce")
        available[name] = col.notna()
        met[name] = (col > threshold) if op == ">" else (col < threshold)
        met[name] = met[name].where(col.notna(), False)

    if met.empty:
        return pd.Series(np.nan, index=features.index, name="authigenic_score")

    count = met.sum(axis=1)
    n_available = available.sum(axis=1)
    if not as_fraction:
        return count.rename("authigenic_score")
    score = count / n_available.where(n_available > 0)
    return score.rename("authigenic_score")


def indicator_detail(features, indicators=None):
    """Per-indicator pass/fail table, for explaining why a sample scored as it did."""
    ind = indicators or AUTHIGENIC_INDICATORS
    out = pd.DataFrame(index=features.index)
    for name, (op, threshold) in ind.items():
        if name in features.columns:
            col = pd.to_numeric(features[name], errors="coerce")
            out[name] = (col > threshold) if op == ">" else (col < threshold)
    return out


def grade_class(df, p2o5_col="P2O5", bands=None):
    """Stage-2 label: assign each sample to a grade band."""
    bands = bands or DEFAULT_GRADE_BANDS
    values = pd.to_numeric(df[p2o5_col], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for low, high, name in bands:
        out = out.mask(values.notna() & (values >= low) & (values < high), name)
    return out.rename("grade_class")


def flag_log_chemistry_conflict(labels, scores, score_cut=0.5):
    """Flag samples where the logged lithology and the geochemistry disagree."""
    labels = labels.astype("float")
    chem_positive = scores >= score_cut
    conflict = pd.Series("agree", index=labels.index, dtype="object")
    conflict = conflict.mask((labels == 0) & chem_positive, "chem_positive_log_barren")
    conflict = conflict.mask((labels == 1) & ~chem_positive, "log_positive_chem_weak")
    return conflict.rename("log_chem_conflict")
