"""Tests for the invariants a resource estimate has to satisfy.

These are not smoke tests. Each one guards a mistake that would silently
produce a plausible-looking but wrong tonnage -- which is the failure mode
that matters in resource estimation, because nothing crashes.

    pytest
"""
from __future__ import annotations

import numpy as np
import pytest

from askot import (
    EstimationParams,
    benchmark_table,
    build_block_model,
    build_grid,
    estimate_resource,
    fit_random_forest,
    grade_tonnage_curve,
    idw_predict,
    load_collars,
    load_composites,
    nearest_sample_distance,
)
from askot import config as C


@pytest.fixture(scope="module")
def composites():
    return load_composites()


@pytest.fixture(scope="module")
def blocks(composites):
    params = EstimationParams()
    model = fit_random_forest(composites)
    return build_block_model(composites, model, params)


# --- data integrity -------------------------------------------------------
def test_composites_have_no_missing_grades(composites):
    assert len(composites) > 20
    assert composites[["X", "Y", "Z", *C.METALS]].notna().all().all()


def test_tmc_is_the_sum_of_the_three_metals(composites):
    expected = composites[list(C.METALS)].sum(axis=1)
    np.testing.assert_allclose(composites["TMC_pct"], expected, rtol=1e-9)


def test_collars_load_on_the_same_grid_as_composites(composites):
    collars = load_collars()
    assert len(collars) > 40
    # Collars and composites must share one coordinate convention, else the
    # 3-D view would show drillholes floating away from the ore they found.
    assert abs(collars["X"].mean() - composites["X"].mean()) < 500
    assert abs(collars["Y"].mean() - composites["Y"].mean()) < 500


# --- geometry -------------------------------------------------------------
def test_grid_covers_composites_plus_padding(composites):
    params = EstimationParams(block_size_m=10.0, padding_m=25.0)
    grid = build_grid(composites, params)
    for i, axis in enumerate(("X", "Y", "Z")):
        assert grid[:, i].min() <= composites[axis].min() - params.padding_m + 1e-9
        assert grid[:, i].max() >= composites[axis].max() - params.block_size_m


def test_smaller_blocks_give_a_denser_grid(composites):
    coarse = build_grid(composites, EstimationParams(block_size_m=20.0))
    fine = build_grid(composites, EstimationParams(block_size_m=5.0))
    assert len(fine) > len(coarse) * 10


def test_nearest_sample_distance_is_zero_at_a_sample():
    samples = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    query = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    got = nearest_sample_distance(query, samples)
    np.testing.assert_allclose(got, [0.0, 0.0, 50.0])


def test_nearest_sample_distance_chunking_matches_single_pass():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(40, 3)) * 50
    query = rng.normal(size=(1200, 3)) * 50
    np.testing.assert_allclose(
        nearest_sample_distance(query, samples, chunk=97),
        nearest_sample_distance(query, samples, chunk=10_000),
    )


# --- estimators -----------------------------------------------------------
def test_idw_reproduces_the_sample_value_at_a_sample_point():
    train_xyz = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    train_y = np.array([[10.0], [2.0]])
    got = idw_predict(train_xyz, train_y, np.array([[0.0, 0.0, 0.0]]))
    assert got[0, 0] == pytest.approx(10.0, abs=1e-3)


def test_idw_output_stays_within_the_sample_range():
    rng = np.random.default_rng(1)
    train_xyz = rng.normal(size=(12, 3)) * 30
    train_y = rng.uniform(1, 9, size=(12, 2))
    got = idw_predict(train_xyz, train_y, rng.normal(size=(50, 3)) * 30)
    # Inverse-distance weights are convex, so predictions can never leave
    # the convex hull of the sample values.
    assert got.min() >= train_y.min() - 1e-9
    assert got.max() <= train_y.max() + 1e-9


def test_random_forest_predicts_three_metals(composites):
    model = fit_random_forest(composites)
    pred = model.predict([[composites["X"].iloc[0],
                           composites["Y"].iloc[0],
                           composites["Z"].iloc[0]]])
    assert pred.shape == (1, 3)
    assert (pred >= 0).all()


# --- the estimate itself --------------------------------------------------
def test_default_run_reproduces_the_published_tonnage(blocks):
    estimate = estimate_resource(blocks, EstimationParams())
    # Published figure is 1.52 Mt; allow a little slack for library drift in
    # the forest's tie-breaking, but flag any real regression.
    assert estimate.tonnage_mt == pytest.approx(1.52, abs=0.08)
    assert 0.77 <= estimate.tonnage_mt <= 1.86, "left the classical envelope"


def test_tonnage_is_blocks_times_volume_times_density(blocks):
    params = EstimationParams(density_t_per_m3=3.5, block_size_m=10.0)
    estimate = estimate_resource(blocks, params)
    assert estimate.tonnage_t == pytest.approx(
        estimate.n_blocks * 1000.0 * 3.5)


def test_tonnage_scales_linearly_with_density(blocks):
    low = estimate_resource(blocks, EstimationParams(density_t_per_m3=3.2))
    high = estimate_resource(blocks, EstimationParams(density_t_per_m3=3.5))
    assert high.tonnage_t / low.tonnage_t == pytest.approx(3.5 / 3.2)


def test_raising_the_cutoff_never_increases_tonnage(blocks):
    tonnages = [estimate_resource(
        blocks, EstimationParams(tmc_cutoff_pct=c)).tonnage_t
        for c in (0.0, 2.0, 5.0, 10.0, 18.0)]
    assert tonnages == sorted(tonnages, reverse=True)


def test_raising_the_cutoff_never_decreases_grade(blocks):
    grades = [estimate_resource(
        blocks, EstimationParams(tmc_cutoff_pct=c)).tmc_pct
        for c in (0.0, 2.0, 5.0, 10.0)]
    assert grades == sorted(grades)


def test_search_radius_is_what_keeps_tonnage_credible(blocks):
    tight = estimate_resource(blocks, EstimationParams(
        max_distance_to_sample_m=15.0))
    loose = estimate_resource(blocks, EstimationParams(
        max_distance_to_sample_m=500.0))
    # Without the radius filter the forest extrapolates ore-like grade
    # across the whole padded box. This is the single biggest source of
    # overstatement in naive ML resource estimation.
    assert loose.tonnage_t > tight.tonnage_t * 5


# --- reporting ------------------------------------------------------------
def test_grade_tonnage_curve_is_monotonic(blocks):
    curve = grade_tonnage_curve(blocks, EstimationParams())
    assert curve["tonnage_t"].is_monotonic_decreasing
    graded = curve.dropna(subset=["TMC_pct"])
    assert graded["TMC_pct"].is_monotonic_increasing


def test_benchmark_table_appends_the_model_row(blocks):
    estimate = estimate_resource(blocks, EstimationParams())
    table = benchmark_table(estimate)
    assert len(table) == len(C.CLASSICAL_ESTIMATES) + 1
    assert table["name"].iloc[-1].startswith("ML model")
    assert table["tonnage_mt"].iloc[-1] == pytest.approx(estimate.tonnage_mt)


def test_benchmark_table_without_a_model_row():
    table = benchmark_table(None)
    assert len(table) == len(C.CLASSICAL_ESTIMATES)
