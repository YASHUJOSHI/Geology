"""Grade estimators and spatial cross-validation.

Two estimators are compared on identical folds:

* **Random Forest** -- multi-output regression of (Cu%, Pb%, Zn%) on
  (X, Y, Z). Non-parametric, so it needs no variogram and no assumption of
  stationarity, which matters for a deposit with only a few dozen assays.
* **Inverse distance weighting** -- the classical interpolator, used as the
  baseline the ML model has to beat to justify itself.

Validation is **leave-one-borehole-out**: every composite from a hole is
held out together. Plain k-fold would put intervals from the same hole in
both train and test, leaking position information and reporting an
optimistic score that would not survive on a new drillhole.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from . import config as C


def _xyz(df: pd.DataFrame) -> np.ndarray:
    return df[["X", "Y", "Z"]].to_numpy(dtype=float)


def _grades(df: pd.DataFrame) -> np.ndarray:
    return df[list(C.METALS)].to_numpy(dtype=float)


def fit_random_forest(composites: pd.DataFrame,
                      rf_params: C.RFParams | None = None) -> RandomForestRegressor:
    """Fit the multi-output Random Forest on every usable composite."""
    rf_params = rf_params or C.RFParams()
    model = RandomForestRegressor(**rf_params.as_kwargs())
    model.fit(_xyz(composites), _grades(composites))
    return model


def idw_predict(train_xyz: np.ndarray, train_y: np.ndarray,
                query_xyz: np.ndarray, power: float = 2.0,
                eps: float = 1e-6) -> np.ndarray:
    """Inverse-distance-weighted grade at each query point.

    ``eps`` keeps the weight finite when a query point coincides with a
    sample. ``power=2`` is the conventional exponent for grade estimation.
    """
    dist = np.linalg.norm(train_xyz[None, :, :] - query_xyz[:, None, :], axis=2)
    weights = 1.0 / np.power(dist + eps, power)
    return (weights @ train_y) / weights.sum(axis=1, keepdims=True)


@dataclass
class CVResult:
    """Per-metal skill scores from leave-one-borehole-out validation."""

    scores: pd.DataFrame
    n_composites: int
    n_holes: int
    n_folds: int

    def summary_line(self) -> str:
        return (f"{self.n_composites} composites from {self.n_holes} holes, "
                f"{self.n_folds} usable folds")


def leave_one_borehole_out_cv(composites: pd.DataFrame,
                              rf_params: C.RFParams | None = None) -> CVResult:
    """Score RF against IDW, holding out one borehole at a time."""
    rf_params = rf_params or C.RFParams()
    xyz, y = _xyz(composites), _grades(composites)
    groups = composites["borehole_id"].to_numpy()

    rf_pred, idw_pred, truth = [], [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(xyz, y, groups):
        # A fold with almost no training data says nothing useful about skill.
        if len(train_idx) < 3:
            continue
        rf = RandomForestRegressor(**rf_params.as_kwargs())
        rf.fit(xyz[train_idx], y[train_idx])
        rf_pred.append(rf.predict(xyz[test_idx]))
        idw_pred.append(idw_predict(xyz[train_idx], y[train_idx], xyz[test_idx]))
        truth.append(y[test_idx])

    n_folds = len(truth)
    rf_pred, idw_pred, truth = (np.vstack(a) for a in (rf_pred, idw_pred, truth))

    rows = []
    for i, metal in enumerate(C.METALS):
        rows.append({
            "metal": metal.replace("_pct", ""),
            "RF_MAE": mean_absolute_error(truth[:, i], rf_pred[:, i]),
            "RF_R2": r2_score(truth[:, i], rf_pred[:, i]),
            "IDW_MAE": mean_absolute_error(truth[:, i], idw_pred[:, i]),
            "IDW_R2": r2_score(truth[:, i], idw_pred[:, i]),
        })

    return CVResult(
        scores=pd.DataFrame(rows),
        n_composites=len(composites),
        n_holes=int(pd.unique(groups).size),
        n_folds=n_folds,
    )


def feature_importance(composites: pd.DataFrame,
                       rf_params: C.RFParams | None = None) -> pd.DataFrame:
    """Gini importance of X/Y/Z, averaged over the three metal targets.

    Fitted one metal at a time because scikit-learn does not expose
    per-output importances from a multi-output forest.
    """
    rf_params = rf_params or C.RFParams()
    xyz, y = _xyz(composites), _grades(composites)

    per_metal = []
    for i in range(len(C.METALS)):
        rf = RandomForestRegressor(**rf_params.as_kwargs())
        rf.fit(xyz, y[:, i])
        per_metal.append(rf.feature_importances_)

    return pd.DataFrame({
        "axis": ["X (departure)", "Y (latitude)", "Z (RL)"],
        "importance": np.mean(per_metal, axis=0),
    })
