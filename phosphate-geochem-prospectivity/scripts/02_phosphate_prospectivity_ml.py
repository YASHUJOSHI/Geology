"""Phosphate prospectivity demo: geochemistry -> CLR -> Random Forest / XGBoost ->
spatial block cross-validation -> interpretation.

Question asked of the model: can the PATHFINDER suite alone (REE, Y, U, Sr, redox and
detrital elements - with P2O5, CaO and F deliberately withheld) recognise samples that
a geologist logged as phosphatic? That is the situation with older multi-element
datasets where P was not reported, and it stops the model from simply reading grade.

Everything runs on SYNTHETIC data (``phosprospect.synthetic``). The point is the
workflow and the validation design, not the numbers.

    python scripts/02_phosphate_prospectivity_ml.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phosprospect import (authigenic_score, build_features, clr_transform,
                          flag_log_chemistry_conflict, label_from_lithology)
from phosprospect.reference import REE_ALL
from phosprospect.synthetic import MAJORS, TRACES, _smooth_field, make_basin

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost is optional; the demo still runs with Random Forest.
    XGBClassifier = None

FIG = ROOT / "figures"
SEED = 0
WITHHELD = ["P2O5", "CaO", "F"]          # direct grade analytes, kept out of the model
BLOCK_SIZE = (1500.0, 1000.0)            # spatial CV block, metres (E, N)
N_FOLDS = 5


# --------------------------------------------------------------------------- data
def composition_ppm(df):
    """All analytes on one scale (ppm) so the log-ratios are meaningful."""
    comp = df[MAJORS + TRACES + REE_ALL].copy()
    comp[MAJORS] = comp[MAJORS] * 10_000.0
    return comp


def make_features(df):
    """CLR-transform the pathfinder composition (closure removed, grade withheld)."""
    comp = composition_ppm(df).drop(columns=WITHHELD)
    return clr_transform(comp)


def spurious_layers(df, n=6, seed=7):
    """Smooth random 'evidence layers' with NO geological link to phosphate.

    Stand-ins for gridded map layers (geophysics, interpolated geochemistry, distance
    buffers). Any skill a model finds in them is location memorisation - exactly what
    random cross-validation rewards and spatial cross-validation exposes.
    """
    rng = np.random.default_rng(seed)
    e, n_ = df["easting"].to_numpy(), df["northing"].to_numpy()
    return pd.DataFrame({f"layer_{i}": _smooth_field(e, n_, rng) for i in range(n)},
                        index=df.index)


def spatial_blocks(df):
    bx = (df["easting"] // BLOCK_SIZE[0]).astype(int)
    by = (df["northing"] // BLOCK_SIZE[1]).astype(int)
    return (bx.astype(str) + "_" + by.astype(str)).rename("block")


# --------------------------------------------------------------------------- models
def models():
    out = {"Random Forest": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=SEED)}
    if XGBClassifier is not None:
        out["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    return out


def cv_splits(scheme, X, y, df):
    if scheme == "random k-fold (samples)":
        return StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)
    if scheme == "grouped by drillhole":
        return GroupKFold(N_FOLDS).split(X, y, groups=df["hole_id"])
    if scheme == "spatial blocks":
        return GroupKFold(N_FOLDS).split(X, y, groups=spatial_blocks(df))
    raise ValueError(scheme)


def out_of_fold(make_model, scheme, X, y, df):
    """Out-of-fold probabilities plus per-fold scores."""
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for train, test in cv_splits(scheme, X, y, df):
        m = make_model().fit(X.iloc[train], y.iloc[train])
        oof[test] = m.predict_proba(X.iloc[test])[:, 1]
        if y.iloc[test].nunique() == 2:
            fold_auc.append(roc_auc_score(y.iloc[test], oof[test]))
    return oof, np.array(fold_auc)


# --------------------------------------------------------------------------- steps
def compare_validation(X, y, df):
    """Same model, three ways of splitting the data. The gap is the lesson."""
    schemes = ["random k-fold (samples)", "grouped by drillhole", "spatial blocks"]
    feature_sets = [("geochemistry (CLR)", X),
                    ("geochemistry + coordinates",
                     X.assign(easting=df["easting"], northing=df["northing"])),
                    ("spurious smooth map layers", spurious_layers(df))]
    rows, oofs = [], {}
    for name, make in models().items():
        for feat_name, feats in feature_sets:
            for scheme in schemes:
                oof, fold_auc = out_of_fold(make, scheme, feats, y, df)
                rows.append({"model": name, "features": feat_name, "validation": scheme,
                             "ROC_AUC": roc_auc_score(y, oof),
                             "PR_AUC": average_precision_score(y, oof),
                             "fold_AUC_sd": fold_auc.std()})
                oofs[(name, feat_name, scheme)] = oof
    res = pd.DataFrame(rows).round(3)
    res.to_csv(FIG / "ml_validation_comparison.csv", index=False)
    print("Cross-validation comparison (higher = better; watch the gaps)\n")
    print(res.to_string(index=False), "\n")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    sub = res[res.model == "Random Forest"]
    labels = sub["features"].unique()
    w = 0.26
    for i, scheme in enumerate(schemes):
        vals = sub[sub.validation == scheme].set_index("features").loc[labels, "ROC_AUC"]
        ax.bar(np.arange(len(labels)) + (i - 1) * w, vals, w, label=scheme)
        for j, v in enumerate(vals):
            ax.text(j + (i - 1) * w, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.text(len(labels) - 0.5, 0.505, "no skill", ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.0)
    ax.set(ylabel="ROC AUC (out-of-fold)",
           title="Random Forest: how the validation design changes the score")
    ax.legend(fontsize=8, loc="lower left")
    stamp(fig)
    fig.tight_layout()
    fig.savefig(FIG / "ml_validation_comparison.png", dpi=150)
    plt.close(fig)
    return res, oofs


def prospectivity_map(df, y, oof):
    """Hole-level out-of-fold probability next to the logged truth."""
    holes = (df.assign(p=oof, y=y.values)
               .groupby("hole_id")
               .agg(easting=("easting", "first"), northing=("northing", "first"),
                    p=("p", "mean"), logged=("y", "mean")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, col, title in [(axes[0], "p", "Model: mean out-of-fold P(phosphatic)\n"
                                          "(spatial-block CV, each hole predicted unseen)"),
                           (axes[1], "logged", "Logged: fraction of samples phosphatic")]:
        sc = ax.scatter(holes.easting, holes.northing, c=holes[col], cmap="YlOrRd",
                        vmin=0, vmax=1, s=60, edgecolor="k", linewidth=0.3)
        for x in np.arange(0, holes.easting.max() + BLOCK_SIZE[0], BLOCK_SIZE[0]):
            ax.axvline(x, color="grey", lw=0.5, ls=":")
        for yb in np.arange(0, holes.northing.max() + BLOCK_SIZE[1], BLOCK_SIZE[1]):
            ax.axhline(yb, color="grey", lw=0.5, ls=":")
        ax.set(title=title, xlabel="easting (m, local)", aspect="equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    axes[0].set_ylabel("northing (m, local)")
    stamp(fig, "dotted lines = CV blocks")
    fig.tight_layout()
    fig.savefig(FIG / "prospectivity_map.png", dpi=150)
    plt.close(fig)
    return holes


def interpret(X, y, df):
    """Permutation importance measured on spatially held-out blocks."""
    imps = []
    for train, test in cv_splits("spatial blocks", X, y, df):
        m = models()["Random Forest"]().fit(X.iloc[train], y.iloc[train])
        r = permutation_importance(m, X.iloc[test], y.iloc[test], scoring="roc_auc",
                                   n_repeats=5, random_state=SEED, n_jobs=-1)
        imps.append(r.importances_mean)
    imp = pd.DataFrame(imps, columns=X.columns)
    summary = pd.DataFrame({"mean": imp.mean(), "sd": imp.std()}).sort_values("mean")
    top = summary.tail(15)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index.str.replace("_clr", ""), top["mean"], xerr=top["sd"],
            color="#c0392b", alpha=0.8)
    ax.set(xlabel="drop in ROC AUC when shuffled (held-out blocks)",
           title="What the model relies on (permutation importance)")
    stamp(fig)
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close(fig)
    summary.sort_values("mean", ascending=False).round(4).to_csv(
        FIG / "feature_importance.csv")
    print("Top pathfinders (CLR):", ", ".join(
        summary.tail(8).index[::-1].str.replace("_clr", "")), "\n")


def log_vs_chemistry(df, y, oof):
    """Cross-check the model and the rule-based indicator score against the core log."""
    score = authigenic_score(build_features(df))
    conflict = flag_log_chemistry_conflict(y, score)
    model_flag = pd.Series(np.where((y == 0) & (oof >= 0.8), "model_positive_log_barren",
                                    np.where((y == 1) & (oof <= 0.2),
                                             "model_negative_log_phosphatic", "agree")),
                           index=df.index)
    out = df[["sample_id", "hole_id", "lithology", "P2O5", "f_francolite_true"]].assign(
        p_model=oof.round(3), indicator_score=score.round(2),
        rule_check=conflict, model_check=model_flag)
    out[["P2O5", "f_francolite_true"]] = out[["P2O5", "f_francolite_true"]].round(3)
    review = out[(out.model_check != "agree")].sort_values("p_model", ascending=False)
    review.to_csv(FIG / "samples_to_relog.csv", index=False)
    print("Indicator-score vs log :", conflict.value_counts().to_dict())
    print("Model vs log           :", model_flag.value_counts().to_dict())
    print(f"{len(review)} samples flagged for re-logging -> figures/samples_to_relog.csv")
    # Only possible because the data are synthetic: were the flags right?
    mislogged = ((review.f_francolite_true >= 0.2) != (review.lithology.str.contains("hosph")))
    print(f"of which genuinely mis-logged (known only in synthetic data): "
          f"{mislogged.sum()} / {len(review)}\n")


def stamp(fig, extra=""):
    note = "SYNTHETIC DEMO DATA - not real samples"
    fig.text(0.99, 0.01, f"{extra}   {note}" if extra else note, ha="right",
             va="bottom", fontsize=7, color="grey", style="italic")


def main():
    FIG.mkdir(exist_ok=True)
    df = make_basin()
    y = label_from_lithology(df)
    X = make_features(df)
    print(f"{len(df)} samples, {df.hole_id.nunique()} holes, "
          f"{spatial_blocks(df).nunique()} spatial blocks, {X.shape[1]} CLR features, "
          f"{y.mean():.0%} logged phosphatic. Withheld from model: {', '.join(WITHHELD)}\n")

    _, oofs = compare_validation(X, y, df)
    best = "XGBoost" if XGBClassifier is not None else "Random Forest"
    oof = oofs[(best, "geochemistry (CLR)", "spatial blocks")]
    prospectivity_map(df, y, oof)
    interpret(X, y, df)
    log_vs_chemistry(df, y, oof)
    print("Figures and tables written to", FIG)


if __name__ == "__main__":
    main()
