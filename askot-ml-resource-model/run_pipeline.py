"""Command-line entry point: fit, validate, estimate, benchmark.

    python run_pipeline.py
    python run_pipeline.py --cutoff 4.0 --block-size 5 --density 3.2
    python run_pipeline.py --skip-cv --write-blocks

Writes the block model and the benchmark comparison to ``data/processed/``.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from askot import config as C
from askot import (
    EstimationParams,
    benchmark_table,
    build_block_model,
    estimate_resource,
    feature_importance,
    fit_random_forest,
    grade_tonnage_curve,
    leave_one_borehole_out_cv,
    load_composites,
    section_weighted_grade,
)
from askot.data import load_sections


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cutoff", type=float, default=EstimationParams.tmc_cutoff_pct,
                   help="TMC cut-off in %% (default: %(default)s)")
    p.add_argument("--block-size", type=float, default=EstimationParams.block_size_m,
                   help="cubic block edge in m (default: %(default)s)")
    p.add_argument("--radius", type=float,
                   default=EstimationParams.max_distance_to_sample_m,
                   help="max block-to-sample distance in m (default: %(default)s)")
    p.add_argument("--density", type=float, default=EstimationParams.density_t_per_m3,
                   help="bulk density in t/m3 (default: %(default)s)")
    p.add_argument("--skip-cv", action="store_true",
                   help="skip leave-one-borehole-out validation (much faster)")
    p.add_argument("--write-blocks", action="store_true",
                   help="write the full block model CSV (~3 MB at 10 m blocks)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    params = EstimationParams(
        tmc_cutoff_pct=args.cutoff,
        block_size_m=args.block_size,
        max_distance_to_sample_m=args.radius,
        density_t_per_m3=args.density,
    )

    print("[1/5] loading composites")
    composites = load_composites()
    print(f"      {len(composites)} composites with usable Cu/Pb/Zn grades "
          f"from {composites['borehole_id'].nunique()} holes")

    if args.skip_cv:
        print("[2/5] cross-validation skipped (--skip-cv)")
    else:
        print("[2/5] leave-one-borehole-out cross-validation (RF vs IDW)")
        cv = leave_one_borehole_out_cv(composites)
        print(f"      {cv.summary_line()}")
        print(cv.scores.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        print("      n is small -- read R2 as relative skill, not precision")

    print("[3/5] fitting Random Forest on all composites")
    model = fit_random_forest(composites)
    print(feature_importance(composites).to_string(index=False,
                                                   float_format=lambda v: f"{v:6.3f}"))

    print(f"[4/5] building block model at {params.block_size_m:g} m blocks")
    blocks = build_block_model(composites, model, params)
    estimate = estimate_resource(blocks, params)
    rejected = estimate.n_blocks_total - estimate.n_blocks_in_range
    print(f"      {estimate.n_blocks_total:,} blocks total; "
          f"{rejected:,} rejected beyond the "
          f"{params.max_distance_to_sample_m:g} m search radius")
    print(f"      {estimate.n_blocks:,} blocks above "
          f"{params.tmc_cutoff_pct:g}% TMC")
    print(f"      tonnage : {estimate.tonnage_t:,.0f} t "
          f"({estimate.tonnage_mt:.2f} Mt) at {params.density_t_per_m3:g} t/m3")
    print(f"      grade   : Cu {estimate.grades['Cu_pct']:.2f}%  "
          f"Pb {estimate.grades['Pb_pct']:.2f}%  "
          f"Zn {estimate.grades['Zn_pct']:.2f}%  "
          f"(TMC {estimate.tmc_pct:.2f}%)")

    sections = section_weighted_grade(load_sections())
    if sections:
        print(f"      independent check -- 1992 cross-sections "
              f"({sections['n_intercepts']} intercepts, true-thickness "
              f"weighted): Cu {sections['Cu_pct']:.2f}%  "
              f"Pb {sections['Pb_pct']:.2f}%  Zn {sections['Zn_pct']:.2f}%")

    print("[5/5] benchmarking against the classical estimates")
    table = benchmark_table(estimate)
    cols = ["name", "tonnage_mt", "Cu_pct", "Pb_pct", "Zn_pct", "cutoff", "method"]
    with pd.option_context("display.max_colwidth", 40):
        print(table[cols].to_string(index=False, na_rep="n/a",
                                    float_format=lambda v: f"{v:6.2f}"))

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.COMPARISON_CSV, index=False)
    print(f"      wrote {C.COMPARISON_CSV.relative_to(C.ROOT)}")

    curve = grade_tonnage_curve(blocks, params)
    curve_path = C.DATA_PROCESSED / "askot_grade_tonnage_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"      wrote {curve_path.relative_to(C.ROOT)}")

    if args.write_blocks:
        blocks.round(3).to_csv(C.BLOCK_MODEL_CSV, index=False)
        print(f"      wrote {C.BLOCK_MODEL_CSV.relative_to(C.ROOT)}")

    print("\nRead the limitations in README.md before quoting any number above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
