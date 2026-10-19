"""Block model construction, search-radius filtering and tonnage reporting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config as C


def build_grid(composites: pd.DataFrame, params: C.EstimationParams) -> np.ndarray:
    """Regular cubic grid over the composite bounding box plus padding.

    Returns an ``(n_blocks, 3)`` array of block-centre coordinates.
    """
    xyz = composites[["X", "Y", "Z"]].to_numpy(dtype=float)
    pad, size = params.padding_m, params.block_size_m

    axes = []
    for i in range(3):
        lo = xyz[:, i].min() - pad
        hi = xyz[:, i].max() + pad
        axes.append(np.arange(lo, hi, size))

    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def nearest_sample_distance(grid_xyz: np.ndarray, sample_xyz: np.ndarray,
                            chunk: int = 5000) -> np.ndarray:
    """Distance from each block centre to its closest composite sample.

    Chunked so the intermediate (n_blocks x n_samples) distance matrix never
    has to exist all at once -- at 5 m blocks this grid runs to ~500k rows.
    """
    out = np.empty(len(grid_xyz))
    for i in range(0, len(grid_xyz), chunk):
        block = grid_xyz[i:i + chunk]
        dist = np.linalg.norm(block[:, None, :] - sample_xyz[None, :, :], axis=2)
        out[i:i + chunk] = dist.min(axis=1)
    return out


def build_block_model(composites: pd.DataFrame, model: RandomForestRegressor,
                      params: C.EstimationParams) -> pd.DataFrame:
    """Predict grades onto the block grid and tag each block's data support.

    The returned frame carries every block, above and below cut-off, plus
    ``dist_to_sample_m``. Keeping the rejected blocks is deliberate: the
    dashboard renders them as the faint halo around the ore shell so the
    viewer can see how much of the modelled volume is being discarded.
    """
    grid = build_grid(composites, params)
    pred = model.predict(grid)
    dist = nearest_sample_distance(grid, composites[["X", "Y", "Z"]].to_numpy(float))

    blocks = pd.DataFrame(grid, columns=["X", "Y", "Z"])
    for i, metal in enumerate(C.METALS):
        blocks[metal] = pred[:, i]
    blocks["TMC_pct"] = pred.sum(axis=1)
    blocks["dist_to_sample_m"] = dist
    return blocks


@dataclass
class ResourceEstimate:
    """One tonnage-and-grade answer, with the parameters that produced it."""

    n_blocks: int
    tonnage_t: float
    grades: dict[str, float]
    tmc_pct: float
    params: C.EstimationParams
    n_blocks_in_range: int
    n_blocks_total: int

    @property
    def tonnage_mt(self) -> float:
        return self.tonnage_t / 1e6

    def as_row(self, name: str = "ML model (Random Forest)") -> dict:
        return dict(
            name=name,
            tonnage_t=self.tonnage_t,
            Cu_pct=self.grades.get("Cu_pct"),
            Pb_pct=self.grades.get("Pb_pct"),
            Zn_pct=self.grades.get("Zn_pct"),
            cutoff=f"{self.params.tmc_cutoff_pct:g}% TMC",
            density=self.params.density_t_per_m3,
            method="Random Forest block model",
        )


def in_range_mask(blocks: pd.DataFrame, params: C.EstimationParams) -> pd.Series:
    """Blocks with a composite sample close enough to be credibly estimated."""
    return blocks["dist_to_sample_m"] <= params.max_distance_to_sample_m


def estimate_resource(blocks: pd.DataFrame,
                      params: C.EstimationParams) -> ResourceEstimate:
    """Tonnage and average grade above cut-off, inside the search radius."""
    supported = in_range_mask(blocks, params)
    ore = blocks[supported & (blocks["TMC_pct"] >= params.tmc_cutoff_pct)]

    volume = len(ore) * params.block_volume_m3()
    tonnage = volume * params.density_t_per_m3
    grades = {m: float(ore[m].mean()) for m in C.METALS} if len(ore) else {
        m: float("nan") for m in C.METALS}

    return ResourceEstimate(
        n_blocks=len(ore),
        tonnage_t=tonnage,
        grades=grades,
        tmc_pct=sum(grades.values()),
        params=params,
        n_blocks_in_range=int(supported.sum()),
        n_blocks_total=len(blocks),
    )


def grade_tonnage_curve(blocks: pd.DataFrame, params: C.EstimationParams,
                        cutoffs: np.ndarray | None = None) -> pd.DataFrame:
    """Classic grade-tonnage relationship: tonnage falls as cut-off rises.

    This is the sanity check a reviewer applies first. A credible estimator
    produces a monotonically falling tonnage curve against a rising grade
    curve; anything else means the grade model is not spatially coherent.
    """
    if cutoffs is None:
        cutoffs = np.arange(0.0, 24.5, 0.5)

    supported = blocks[in_range_mask(blocks, params)]
    tmc = supported["TMC_pct"].to_numpy()
    volume_factor = params.block_volume_m3() * params.density_t_per_m3

    rows = []
    for cut in cutoffs:
        ore = supported[tmc >= cut]
        rows.append({
            "cutoff_tmc_pct": float(cut),
            "blocks": len(ore),
            "tonnage_t": len(ore) * volume_factor,
            "tonnage_mt": len(ore) * volume_factor / 1e6,
            "Cu_pct": float(ore["Cu_pct"].mean()) if len(ore) else np.nan,
            "Pb_pct": float(ore["Pb_pct"].mean()) if len(ore) else np.nan,
            "Zn_pct": float(ore["Zn_pct"].mean()) if len(ore) else np.nan,
            "TMC_pct": float(ore["TMC_pct"].mean()) if len(ore) else np.nan,
        })
    return pd.DataFrame(rows)


def benchmark_table(estimate: ResourceEstimate | None = None) -> pd.DataFrame:
    """Classical estimates, optionally with this model's answer appended."""
    rows = [dict(r) for r in C.CLASSICAL_ESTIMATES]
    if estimate is not None:
        rows.append(estimate.as_row())
    df = pd.DataFrame(rows)
    df["tonnage_mt"] = df["tonnage_t"] / 1e6
    return df
