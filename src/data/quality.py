"""Reusable data-quality checks for the Customer Intelligence Platform.

Every function takes a DataFrame (or Series) and returns a small, JSON-serialisable
result — a dict or a tidy DataFrame — so checks can be run standalone, combined into
the report built by ``scripts/run_data_quality.py``, or asserted on in tests.

Nothing in this module mutates its input or drops rows. Deciding what to do about a
finding is a separate, explicit step (see ``clean_customer_features`` at the bottom),
so a quality check can never silently change the data underneath the caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, worst first."""
    n_missing = df.isna().sum()
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": (n_missing / len(df) * 100).round(2)})
    return report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, Any]:
    """Exact duplicate rows overall, and on a business key if one is given."""
    result: dict[str, Any] = {"n_exact_duplicate_rows": int(df.duplicated().sum())}
    if subset is not None:
        key_dupes = df.duplicated(subset=subset)
        result[f"n_duplicate_on_{'_'.join(subset)}"] = int(key_dupes.sum())
    return result


def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    """Declared dtype next to the dtype pandas would infer from the values.

    A mismatch (e.g. a numeric ID stored as float because of NaNs, or a date
    stored as text) is exactly the kind of silent problem this is meant to surface.
    """
    rows = []
    for col in df.columns:
        declared = str(df[col].dtype)
        non_null = df[col].dropna()
        inferred = pd.api.types.infer_dtype(non_null, skipna=True) if len(non_null) else "empty"
        rows.append({"column": col, "declared_dtype": declared, "inferred_content": inferred})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Value-level checks
# ---------------------------------------------------------------------------


def impossible_value_report(df: pd.DataFrame, rules: dict[str, str]) -> pd.DataFrame:
    """Count violations of caller-supplied validity rules.

    ``rules`` maps a column name to a pandas query expression that describes an
    IMPOSSIBLE value for that column, e.g. ``{"quantity": "quantity == 0"}``.
    Kept generic and rule-driven rather than hard-coded, since what counts as
    "impossible" depends on the table (a negative quantity is a return in this
    dataset, not an error).
    """
    rows = []
    for column, expr in rules.items():
        try:
            n_violations = int(df.eval(expr).sum())
        except Exception as exc:  # noqa: BLE001 - surface the bad rule, don't crash the report
            rows.append({"column": column, "rule": expr, "violations": None, "error": str(exc)})
            continue
        rows.append({"column": column, "rule": expr, "violations": n_violations, "error": None})
    return pd.DataFrame(rows)


def category_consistency_report(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]:
    """Value counts per categorical column, for spotting inconsistent labelling
    (near-duplicate spellings, placeholder values like 'Unspecified', mixed case).
    """
    return {col: df[col].value_counts(dropna=False) for col in columns}


def date_consistency_report(
    series: pd.Series, min_date: str | pd.Timestamp, max_date: str | pd.Timestamp
) -> dict[str, Any]:
    """Rows outside a known-valid date range, and any non-monotonic surprises."""
    ts = pd.to_datetime(series)
    lo, hi = pd.Timestamp(min_date), pd.Timestamp(max_date)
    return {
        "min_observed": ts.min(),
        "max_observed": ts.max(),
        "n_before_expected_range": int((ts < lo).sum()),
        "n_after_expected_range": int((ts > hi).sum()),
        "n_null": int(ts.isna().sum()),
    }


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """describe() plus skewness and a negative-value count, for the columns given."""
    desc = df[columns].describe().T
    desc["skew"] = df[columns].skew()
    desc["n_negative"] = (df[columns] < 0).sum()
    desc["n_zero"] = (df[columns] == 0).sum()
    return desc.round(3)


def iqr_outlier_report(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    """Tukey IQR fences: how many values fall outside [Q1 - k*IQR, Q3 + k*IQR].

    Flags outliers for review — does not remove them. In a dataset with resellers
    mixed among consumers (this one), a statistical outlier is often a real
    high-volume customer, not an error.
    """
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append(
            {
                "column": col,
                "lower_fence": round(lo, 2),
                "upper_fence": round(hi, 2),
                "n_outliers": n_out,
                "pct_outliers": round(n_out / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Target / leakage checks
# ---------------------------------------------------------------------------


def target_distribution(y: pd.Series) -> pd.DataFrame:
    """Class counts and the imbalance ratio (majority : minority)."""
    counts = y.value_counts()
    pct = (counts / len(y) * 100).round(2)
    ratio = round(counts.max() / counts.min(), 2) if counts.min() > 0 else float("inf")
    out = pd.DataFrame({"count": counts, "pct": pct})
    out.attrs["imbalance_ratio"] = ratio
    return out


def correlation_with_target(
    df: pd.DataFrame, target: str, columns: list[str] | None = None, flag_threshold: float = 0.6
) -> pd.DataFrame:
    """Point-biserial correlation of each numeric feature with a binary target.

    A very high correlation (above ``flag_threshold``) is not proof of leakage —
    it can be a legitimately strong predictor — but it is exactly the signal that
    should trigger a manual check of *how* the feature was computed relative to
    the label's observation window before it is trusted.
    """
    columns = columns or [c for c in df.select_dtypes("number").columns if c != target]
    corr = df[columns].corrwith(df[target].astype(float)).rename("corr_with_target")
    out = corr.to_frame()
    out["abs_corr"] = out["corr_with_target"].abs()
    out["flagged_for_review"] = out["abs_corr"] >= flag_threshold
    return out.sort_values("abs_corr", ascending=False)


def constant_or_near_constant_columns(df: pd.DataFrame, threshold: float = 0.99) -> list[str]:
    """Columns where one value accounts for >= ``threshold`` of rows.

    Such a feature carries almost no information and is a candidate for removal
    for a reason unrelated to leakage (it just can't help the model).
    """
    flagged = []
    for col in df.columns:
        top_share = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_share >= threshold:
            flagged.append(col)
    return flagged


def highly_correlated_pairs(df: pd.DataFrame, columns: list[str], threshold: float = 0.95) -> pd.DataFrame:
    """Numeric column pairs correlated above ``threshold`` — duplicated information
    that inflates dimensionality and destabilises coefficient-based models without
    adding predictive signal.
    """
    corr = df[columns].corr().abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .rename("abs_corr")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False)


# ---------------------------------------------------------------------------
# Cleaning — explicit, documented, separate from detection
# ---------------------------------------------------------------------------


def clean_customer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the treatments decided in reports/data_quality_report.md.

    Returns the cleaned frame AND a log of what was changed and why, so the
    transformation is auditable rather than a black box. Every treatment here
    adds information (a flag, an imputed value) rather than deleting rows —
    the customer count must stay bandwidth-identical to the input so that
    ``churn_labels`` and the feature table never fall out of sync.
    """
    out = df.copy()
    log: list[dict[str, Any]] = []

    # 1,131 single-purchase customers have no interpurchase gap to average,
    # and 1,919 with <3 purchases have no gap variance — this is a structural
    # consequence of low frequency, not missing data, so it is flagged rather
    # than imputed with a value (e.g. 0 or the mean) that would misrepresent
    # a customer who has only ever ordered once.
    for col in ("avg_interpurchase_days", "std_interpurchase_days"):
        if col in out.columns:
            flag_col = f"{col}_is_missing"
            out[flag_col] = out[col].isna()
            log.append(
                {
                    "column": col,
                    "issue": f"{int(out[flag_col].sum())} nulls (customers with too few orders to "
                    "compute a gap)",
                    "treatment": f"kept as null; added boolean flag '{flag_col}' so models can "
                    "distinguish 'no gap exists' from 'gap is zero'",
                }
            )

    # return_rate > 1 means a customer's returns this window exceed their
    # purchases this window (e.g. returning stock bought before the lookback
    # window began). Genuine, not an error — capped for modelling stability,
    # original value kept alongside so nothing is silently lost.
    if "return_rate" in out.columns:
        n_extreme = int((out["return_rate"] > 1).sum())
        if n_extreme:
            out["return_rate_raw"] = out["return_rate"]
            out["return_rate"] = out["return_rate"].clip(upper=1.0)
            log.append(
                {
                    "column": "return_rate",
                    "issue": f"{n_extreme} customers with return_rate > 1.0 (returns exceed "
                    "in-window purchases)",
                    "treatment": "capped at 1.0 for modelling; original preserved in " "'return_rate_raw'",
                }
            )

    return out, log
