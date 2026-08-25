"""Step 4 — Data quality and validation.

Profiles BOTH the raw transactional data and the analytical feature table built
by Step 3, using the reusable checks in src/data/quality.py, and writes a single
Markdown report. Cleaning is explicit and separate from detection: the raw file
is never touched, and the feature table's cleaned copy is written alongside the
original rather than over it.

Run:
    python scripts/run_data_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import CONFIG, PATHS  # noqa: E402
from src.data import quality as q  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402

logger = get_logger(__name__)

FEATURE_FILE = "customer_features_2011-06-09_h183.parquet"
REPORT_PATH = PATHS.reports / "data_quality_report.md"

RAW_NUMERIC_COLS = ["Quantity", "Price"]
RAW_CATEGORY_COLS = ["Country"]

FEATURE_NUMERIC_COLS = [
    "recency_days", "frequency", "monetary_total", "monetary_avg_order",
    "tenure_days", "active_days", "avg_interpurchase_days", "std_interpurchase_days",
    "purchase_rate_per_month", "total_items", "avg_items_per_order", "distinct_products",
    "avg_unit_price", "return_invoices", "return_value", "return_rate",
    "orders_last_30d", "orders_last_90d", "spend_last_90d", "spend_ratio_90d",
]


def profile_raw(raw: pd.DataFrame) -> list[str]:
    sections = ["## 1. Raw transactional data (`data/raw/online_retail_II.csv`)\n"]

    sections.append(
        f"- Rows: **{len(raw):,}**\n"
        f"- Columns: **{raw.shape[1]}**\n"
        f"- Identified customers: **{raw['Customer ID'].nunique():,}**\n"
        f"- Date range: **{raw.InvoiceDate.min()} -> {raw.InvoiceDate.max()}**\n"
    )

    sections.append("### Missing values\n")
    sections.append(md_table(q.missing_value_report(raw)) + "\n")

    sections.append("### Duplicate rows\n")
    dupes = q.duplicate_report(raw)
    same_line = raw.duplicated(
        subset=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID"]
    ).sum()
    sections.append(
        f"- Exact duplicate rows (all 8 columns identical): **{dupes['n_exact_duplicate_rows']:,}**\n"
        f"- Of these, same-invoice/same-line duplicates: **{int(same_line):,}**\n"
        "- **Finding:** inspection shows these are the *same product appearing as separate "
        "line entries on the same invoice with identical quantity, price and timestamp* "
        "(e.g. invoice 489517, stock code 21912 appears 3 times identically). This is a "
        "known characteristic of till/EPOS systems that log each physical pick as its own "
        "line rather than aggregating quantity — not a data-entry error.\n"
        "- **Treatment: KEEP.** Deduplicating would silently understate revenue and item "
        "counts for affected invoices. No row is dropped for this reason anywhere in the "
        "pipeline.\n"
    )

    sections.append("### Impossible / questionable values\n")
    rules = {
        "Quantity": "Quantity == 0",
        "Price": "Price <= 0",
        "InvoiceDate": "InvoiceDate.isna()",
    }
    sections.append(md_table(q.impossible_value_report(raw, rules), index=False) + "\n")
    sections.append(
        f"- Negative `Quantity` (returns/credits): **{(raw.Quantity < 0).sum():,}** "
        "— legitimate (credit-note convention), not an error.\n"
        f"- Negative `Price` (adjustment write-offs): **{(raw.Price < 0).sum():,}** "
        "— confined to the 6 'Adjust bad debt' invoices; excluded from features via "
        "`invoice_type = 'ADJUSTMENT'`.\n"
    )

    sections.append("### Category consistency — `Country`\n")
    counts = q.category_consistency_report(raw, RAW_CATEGORY_COLS)["Country"]
    ambiguous = ["Unspecified", "European Community", "RSA", "Channel Islands"]
    flagged = counts.reindex(ambiguous).dropna().astype(int)
    sections.append(
        f"- {raw.Country.nunique()} distinct country values. Ambiguous or non-ISO labels found:\n\n"
        + md_table(flagged.rename("rows").to_frame())
        + "\n\n- `RSA` = South Africa, `EIRE` = Ireland (both non-standard but internally "
        "consistent, so not merged). `Unspecified` and `European Community` cannot be "
        "resolved to a real country from this data.\n"
        "- **Treatment:** kept as-is; `is_uk` in the feature table gives models the one "
        "country split that matters (98.6% of raw rows are UK). Country-level modelling "
        "beyond that split is not warranted given how concentrated the data is.\n"
    )

    sections.append("### Date consistency\n")
    dates = q.date_consistency_report(raw["InvoiceDate"], "2009-12-01", "2011-12-09")
    sections.append(
        f"- Observed range: {dates['min_observed']} -> {dates['max_observed']}\n"
        f"- Rows outside the expected range: **{dates['n_before_expected_range'] + dates['n_after_expected_range']}**\n"
        f"- Null timestamps: **{dates['n_null']}**\n"
    )

    return sections


def profile_features(fdf: pd.DataFrame) -> list[str]:
    sections = [f"## 2. Analytical feature table (`data/processed/{FEATURE_FILE}`)\n"]

    sections.append(
        f"- Customers (rows): **{len(fdf):,}**\n"
        f"- Columns: **{fdf.shape[1]}**\n"
    )

    sections.append("### Duplicate customers\n")
    dupes = q.duplicate_report(fdf, subset=["customer_id"])
    sections.append(f"- Duplicate `customer_id`: **{dupes['n_duplicate_on_customer_id']}**\n")

    sections.append("### Missing values\n")
    miss = q.missing_value_report(fdf)
    sections.append(
        (md_table(miss) if not miss.empty else "_None._")
        + "\n\n- `avg_interpurchase_days` / `std_interpurchase_days` are null for customers "
        "with too few orders to compute a gap (1 order, or <3 orders respectively) — "
        "**structural, not missing data**. See treatment below.\n"
    )

    sections.append("### Numerical distributions\n")
    sections.append(md_table(q.numeric_summary(fdf, FEATURE_NUMERIC_COLS)) + "\n")

    sections.append("### Outliers (Tukey IQR, k=1.5)\n")
    outliers = q.iqr_outlier_report(fdf, FEATURE_NUMERIC_COLS)
    sections.append(md_table(outliers, index=False) + "\n")
    sections.append(
        "- **Interpretation:** the highest outlier rates are in `monetary_total`, "
        "`frequency` and `total_items` — expected in a dataset that mixes ordinary "
        "consumers with wholesale/reseller accounts. These are not measurement errors "
        "and are **not removed**; tree-based models handle this skew natively, and "
        "Step 7's linear baseline will need a log transform, not row deletion.\n"
    )

    sections.append("### Category distribution — `country_name`\n")
    top_countries = fdf["country_name"].value_counts().head(10).rename("customers").to_frame()
    sections.append(md_table(top_countries) + "\n")

    sections.append("### Target distribution — `is_churned`\n")
    target = q.target_distribution(fdf["is_churned"])
    sections.append(
        md_table(target)
        + f"\n\n- Imbalance ratio (majority:minority): **{target.attrs['imbalance_ratio']}:1**\n"
        "- **Assessment:** 42.5%/57.5% is a MILD imbalance. It does not require SMOTE or "
        "undersampling; class weighting (`class_weight='balanced'`) is sufficient and will "
        "be used from the baseline model onward.\n"
    )

    sections.append("### Correlation with target (leakage screen)\n")
    corr = q.correlation_with_target(fdf, target="is_churned", columns=FEATURE_NUMERIC_COLS, flag_threshold=0.5)
    sections.append(md_table(corr.round(3)) + "\n")
    top = corr.index[0]
    sections.append(
        f"- **`{top}` correlates most strongly with the target ({corr.loc[top, 'corr_with_target']:.3f}).** "
        "This is expected, not leakage: recency is computed strictly from pre-cutoff purchases "
        "(sql/validation.sql checks 11-14 assert this), and churn is *defined* as an absence of "
        "purchases after the cutoff — the two are naturally related without either using the "
        "other's information. It is flagged here so it gets deliberate attention in SHAP "
        "(Step 11) rather than being treated as a surprise.\n"
        "- No feature reads data from after the cutoff; the leakage screen exists to catch a "
        "future feature-engineering mistake, not to re-litigate Step 3.\n"
    )

    sections.append("### Constant / near-constant columns\n")
    const_cols = q.constant_or_near_constant_columns(fdf, threshold=0.99)
    sections.append(
        f"- {', '.join(const_cols) if const_cols else 'None found.'}\n\n"
        "- **Explanation:** `cutoff_date` is constant because this export holds a single "
        "label definition (see the SQL pipeline README section) — it is an artefact of "
        "exporting one cutoff, not a data-quality defect. It carries no predictive "
        "information and should be dropped before modelling.\n"
    )

    sections.append("### Highly correlated feature pairs (duplicate information, |r| >= 0.9)\n")
    pairs = q.highly_correlated_pairs(fdf, FEATURE_NUMERIC_COLS, threshold=0.9)
    sections.append(
        (md_table(pairs.round(3), index=False) if not pairs.empty else "_None found._") + "\n"
    )
    if not pairs.empty:
        sections.append(
            "- `frequency` (orders placed) and `active_days` (distinct purchase days) are "
            "correlated at 0.961 because most customers place at most one order per active "
            "day; they diverge only for customers who raise multiple invoices on the same day.\n"
            "- **Treatment:** not dropped at this stage — kept visible so Step 6/7 can decide "
            "per-model (a linear model benefits from removing one of a correlated pair; a "
            "tree-based model does not need to).\n"
        )

    return sections


def main() -> int:
    raw_path = PATHS.data_raw / CONFIG["data"]["raw_file"]
    feature_path = PATHS.data_processed / FEATURE_FILE

    if not raw_path.is_file():
        logger.error("Raw file not found: %s", raw_path)
        return 1
    if not feature_path.is_file():
        logger.error("Feature file not found: %s (run scripts/run_pipeline.py first)", feature_path)
        return 1

    logger.info("Loading raw data: %s", raw_path)
    raw = pd.read_csv(raw_path, parse_dates=["InvoiceDate"], dtype={"Invoice": "string", "StockCode": "string"})

    logger.info("Loading feature table: %s", feature_path)
    fdf = pd.read_parquet(feature_path)

    logger.info("Running raw-data checks")
    raw_sections = profile_raw(raw)

    logger.info("Running feature-table checks")
    feature_sections = profile_features(fdf)

    logger.info("Applying documented cleaning treatments")
    cleaned, change_log = q.clean_customer_features(fdf)

    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    cleaned_path = PATHS.data_processed / FEATURE_FILE.replace(".parquet", "_validated.parquet")
    cleaned.to_parquet(cleaned_path, index=False)
    logger.info("Wrote cleaned/validated table: %s", cleaned_path.relative_to(PATHS.root))

    assert len(cleaned) == len(fdf), "Cleaning must not change the customer count"
    assert cleaned["customer_id"].equals(fdf["customer_id"]), "Cleaning must not reorder or drop customers"

    treatment_section = ["## 3. Treatments applied\n"]
    if change_log:
        log_df = pd.DataFrame(change_log)
        treatment_section.append(md_table(log_df, index=False) + "\n")
    else:
        treatment_section.append("_No treatments were necessary._\n")
    treatment_section.append(
        f"\nOriginal file is untouched at `data/processed/{FEATURE_FILE}` "
        f"({len(fdf):,} rows). Validated copy at "
        f"`data/processed/{cleaned_path.name}` ({len(cleaned):,} rows, "
        f"{cleaned.shape[1] - fdf.shape[1]} added flag column(s)).\n"
    )

    summary_section = [
        "## Summary — identified problems and recommended treatment\n",
        md_table(
            pd.DataFrame(
                [
                    {"Problem": "34,335 exact duplicate raw line items", "Severity": "Low",
                     "Treatment": "Keep — legitimate repeat line entries, not data-entry errors"},
                    {"Problem": "22.77% of raw rows have no Customer ID", "Severity": "Expected",
                     "Treatment": "Keep in DB (real revenue); excluded from customer-level features by design"},
                    {"Problem": "Ambiguous country labels (Unspecified, European Community, RSA)",
                     "Severity": "Low", "Treatment": "Keep; `is_uk` binary flag is the feature that matters"},
                    {"Problem": "Right-skewed monetary/frequency features with reseller-driven outliers",
                     "Severity": "Medium", "Treatment": "Keep rows; log-transform for linear models (Step 7)"},
                    {"Problem": "return_rate > 1 for some customers", "Severity": "Low",
                     "Treatment": "Capped at 1.0 for modelling; raw value preserved in return_rate_raw"},
                    {"Problem": "Null interpurchase-gap features for low-frequency customers",
                     "Severity": "Expected", "Treatment": "Kept null + added *_is_missing flags"},
                    {"Problem": "recency_days strongly correlated with target", "Severity": "Watch",
                     "Treatment": "Not leakage (pre-cutoff only, verified in SQL); flagged for SHAP scrutiny"},
                    {"Problem": "Mild class imbalance (42.5% / 57.5%)", "Severity": "Low",
                     "Treatment": "class_weight='balanced'; no resampling needed"},
                ]
            ),
            index=False,
        ),
        "",
    ]

    report = [
        "# Data Quality Report",
        "",
        "Generated by `scripts/run_data_quality.py`. All figures are measured from the "
        "actual project data — none are estimated or assumed.",
        "",
        *raw_sections,
        *feature_sections,
        *treatment_section,
        *summary_section,
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
