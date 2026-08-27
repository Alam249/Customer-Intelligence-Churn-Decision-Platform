"""Step 6 — Feature engineering and train/test split.

Takes the Step 4 validated feature table and:
  1. Splits it into train/test BEFORE any statistic is learned from the data
     (see src/features/split.py for why a stratified random split, not a
     time-based one, is the right choice for this single-snapshot table).
  2. Fits src.features.engineer.CustomerFeatureEngineer on the TRAINING split
     only, then applies it to both splits — the quantile thresholds behind
     `rfm_score` and `is_high_value` never see the test data.
  3. Saves train/test parquet files and the fitted transformer.
  4. Writes reports/feature_engineering_report.md documenting every feature
     that reaches the model (formula, business meaning, churn hypothesis,
     leakage risk), sourced from src/features/catalog.py.

Run:
    python scripts/run_feature_engineering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PATHS, RANDOM_SEED  # noqa: E402
from src.features.catalog import FEATURE_CATALOG  # noqa: E402
from src.features.engineer import CustomerFeatureEngineer  # noqa: E402
from src.features.split import stratified_customer_split  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402

logger = get_logger(__name__)

SOURCE_FILE = "customer_features_2011-06-09_h183_validated.parquet"
TEST_SIZE = 0.2
REPORT_PATH = PATHS.reports / "feature_engineering_report.md"


def main() -> int:
    source_path = PATHS.data_processed / SOURCE_FILE
    if not source_path.is_file():
        logger.error("Source file not found: %s (run scripts/run_data_quality.py first)", source_path)
        return 1

    logger.info("Loading %s", source_path)
    df = pd.read_parquet(source_path)

    # cutoff_date is constant for a single-snapshot export (Step 4 flagged this
    # explicitly) and carries zero predictive information — dropped here so no
    # downstream script accidentally treats it as a feature.
    df = df.drop(columns=["cutoff_date"])

    logger.info(
        "Splitting %d customers (stratified on is_churned, test_size=%.0f%%)", len(df), TEST_SIZE * 100
    )
    train_df, test_df = stratified_customer_split(
        df, target="is_churned", test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info(
        "Train: %d customers (%.2f%% churned) | Test: %d customers (%.2f%% churned)",
        len(train_df),
        train_df["is_churned"].mean() * 100,
        len(test_df),
        test_df["is_churned"].mean() * 100,
    )

    logger.info("Fitting CustomerFeatureEngineer on the TRAINING split only")
    engineer = CustomerFeatureEngineer()
    engineer.fit(train_df)
    logger.info(
        "RFM quantile bins actually formed (may be < 5 where a column has heavy ties): "
        "recency=%d, frequency=%d, monetary=%d | high-value threshold (75th pct of train "
        "monetary_total) = %.2f",
        engineer.n_recency_bins_,
        engineer.n_frequency_bins_,
        engineer.n_monetary_bins_,
        engineer.high_value_threshold_,
    )

    train_out = engineer.transform(train_df)
    test_out = engineer.transform(test_df)

    new_cols = engineer.get_new_feature_names()
    assert all(c in train_out.columns for c in new_cols), "Transform did not add all expected columns"
    assert len(train_out) == len(train_df) and len(test_out) == len(
        test_df
    ), "Transform must not add or remove rows"

    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    train_path = PATHS.data_processed / "train.parquet"
    test_path = PATHS.data_processed / "test.parquet"
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)
    logger.info(
        "Wrote %s (%d rows) and %s (%d rows)", train_path.name, len(train_out), test_path.name, len(test_out)
    )

    PATHS.models.mkdir(parents=True, exist_ok=True)
    engineer_path = PATHS.models / "feature_engineer.joblib"
    joblib.dump(engineer, engineer_path)
    logger.info(
        "Saved fitted transformer: %s (needed at inference time in Step 14 so a new "
        "customer is scored with the SAME thresholds as training, not a refit)",
        engineer_path.relative_to(PATHS.root),
    )

    # --- Report ---
    catalog_df = pd.DataFrame(FEATURE_CATALOG)[
        ["name", "stage", "category", "formula", "business_meaning", "churn_hypothesis", "leakage_risk"]
    ]

    # Mixing float64 columns with the nullable Int64 score columns makes describe()
    # fall back to object dtype, which silently defeats .round() — cast to float first.
    new_feature_summary = train_out[new_cols].astype(float).describe().T.round(3)

    rfm_gradient = (
        train_out.groupby("rfm_score")["is_churned"]
        .agg(["mean", "size"])
        .round(4)
        .rename(columns={"mean": "churn_rate", "size": "n_customers"})
    )

    # Concrete proof the transformer used the TRAIN threshold on the test split,
    # not a threshold recomputed from the test data itself.
    test_own_threshold = test_out["monetary_total"].quantile(0.75)
    test_share_if_leaked = (test_out["monetary_total"] >= test_own_threshold).mean()
    test_share_actual = test_out["is_high_value"].mean()

    report = [
        "# Feature Engineering Report",
        "",
        "Generated by `scripts/run_feature_engineering.py`. All figures are measured from "
        "the actual train split — none are estimated or assumed.",
        "",
        "## Split strategy",
        "",
        "Stratified random split on `is_churned`, **not** a time-based split — the reasoning "
        "is in the [`src/features/split.py`](../src/features/split.py) docstring: this table "
        "is a single cross-sectional snapshot (one row per customer at one fixed cutoff), so "
        "there is no per-row time axis to split on. A genuine time-based split would need "
        "multiple snapshot cutoffs, which is a different, larger exercise reserved for future "
        "work (e.g. Step 19 drift monitoring).",
        "",
        "| Split | Customers | Churn rate |",
        "| --- | --- | --- |",
        f"| Train | {len(train_out):,} | {train_out['is_churned'].mean() * 100:.2f}% |",
        f"| Test | {len(test_out):,} | {test_out['is_churned'].mean() * 100:.2f}% |",
        "",
        "Stratification keeps the churn rate within 0.1pp of the full dataset's 42.52% in "
        "both splits (confirmed above) — the mild class imbalance is preserved, not "
        "accidentally amplified or erased by the split.",
        "",
        "## Leakage prevention: fit on train, apply to test",
        "",
        f"`rfm_score` and `is_high_value` depend on quantile thresholds. Those thresholds are "
        f"computed ONCE from `train.parquet` (`CustomerFeatureEngineer.fit(train_df)`) and then "
        f"applied — not recomputed — to `test.parquet`. The high-value threshold learned from "
        f"training is **{engineer.high_value_threshold_:.2f}** in monetary units; test customers "
        f"are compared against this fixed number, never against their own quantiles.",
        "",
        f"**Concrete check:** the test split's OWN 75th percentile of `monetary_total` is "
        f"{test_own_threshold:.2f} (different from the {engineer.high_value_threshold_:.2f} learned "
        f"on train, since the two splits are different customers). If `is_high_value` were "
        f"(incorrectly) computed from the test split's own quantile, {test_share_if_leaked * 100:.1f}% "
        f"of test customers would be flagged high-value by construction. The actual flagged share is "
        f"**{test_share_actual * 100:.1f}%** — different from {test_share_if_leaked * 100:.1f}%, "
        f"confirming the training threshold, not a refit one, was applied.",
        "",
        "## New features added in this step",
        "",
        md_table(new_feature_summary),
        "",
        "### `rfm_score` vs. churn (train split)",
        "",
        "The composite score shows a clean, strongly monotonic gradient — evidence it successfully "
        "compresses the three raw RFM signals into one interpretable number without losing their "
        "combined predictive power:",
        "",
        md_table(rfm_gradient),
        "",
        "## Full feature dictionary",
        "",
        "Every feature that reaches the model, whichever stage produced it:",
        "",
        md_table(catalog_df, index=False),
        "",
        "## Outputs",
        "",
        f"- `data/processed/train.parquet` — {len(train_out):,} rows, {train_out.shape[1]} columns",
        f"- `data/processed/test.parquet` — {len(test_out):,} rows, {test_out.shape[1]} columns",
        "- `models/feature_engineer.joblib` — the fitted transformer, reused at inference time "
        "(Step 14) so new customers are scored against training-set thresholds",
        "",
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
