"""B1 entry point: run EDA, clean the raw data, and write artifacts.

Usage (from the repo root):
    python scripts/01_eda.py

Produces:
    data/cleaned.parquet
    reports/eda/*.csv   (summary tables)
    reports/eda/*.png   (distribution / pattern / correlation plots)
and prints the B1 integrity gate so a failure is obvious in CI or a demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root modules importable when run as `python scripts/01_eda.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cleaning
import config
import eda
import io_utils


def main() -> int:
    """Run the full B1 pipeline. Returns 0 on a passing gate, 1 otherwise."""
    config.EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSV ...")
    raw = io_utils.load_raw()
    print(f"  raw shape: {raw.shape}")

    # Occupancy exceedance must be measured on the RAW frame (pre-cap).
    eda.occupancy_exceedance(raw).to_csv(
        config.EDA_REPORTS_DIR / "occupancy_exceedance.csv", index=False
    )

    print("Cleaning ...")
    df = cleaning.clean(raw)

    print("Writing EDA tables ...")
    eda.missing_report(df).to_csv(
        config.EDA_REPORTS_DIR / "missing_report.csv", index=False
    )
    eda.per_link_congestion(df).to_csv(
        config.EDA_REPORTS_DIR / "per_link_congestion.csv", index=False
    )
    eda.lane6_analysis(df).to_csv(
        config.EDA_REPORTS_DIR / "lane6_analysis.csv", index=False
    )
    eda.hourly_patterns(df).to_csv(
        config.EDA_REPORTS_DIR / "hourly_patterns.csv", index=False
    )
    eda.daily_patterns(df).to_csv(
        config.EDA_REPORTS_DIR / "daily_patterns.csv", index=False
    )
    eda.correlation_matrix(df).to_csv(
        config.EDA_REPORTS_DIR / "correlation_matrix.csv"
    )

    print("Writing EDA plots ...")
    eda.plot_metric_distributions(df, config.EDA_REPORTS_DIR)
    eda.plot_hourly(df, config.EDA_REPORTS_DIR)
    eda.plot_correlation(df, config.EDA_REPORTS_DIR)

    print(f"Saving cleaned parquet -> {config.CLEANED_PARQUET}")
    io_utils.save_parquet(df, config.CLEANED_PARQUET)

    report = cleaning.integrity_report(df)
    print("\n=== B1 INTEGRITY GATE ===")
    for key, value in report.items():
        print(f"  {key:>22}: {value}")

    passed = (
        report["rows_match_expected"]
        and report["missing_cells"] == 0
        and report["max_occupancy"] <= config.OCCUPANCY_CAP
        and report["timeint_dropped"]
        and report["has_lane6_active"]
        and report["has_stall_flags"]
        and report["links"] == config.EXPECTED_LINKS
    )
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
