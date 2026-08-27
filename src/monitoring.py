"""Data and prediction drift detection (Step 19).

Why this exists
----------------
Every prior step measured the model against a test set drawn from the SAME
population it was trained on (a stratified random split of the single
2011-06-09 snapshot — see `scripts/run_feature_engineering.py`). That answers
"does this model work on this data." It says nothing about whether the
customer population the model would score TODAY still resembles the
population it was trained on. This module answers that second question with
two industry-standard statistics, computed feature-by-feature:

  - **Population Stability Index (PSI)**: buckets the reference distribution
    into deciles, then measures what share of the CURRENT population falls
    into each bucket relative to reference. PSI < 0.10 = no material shift,
    0.10-0.25 = moderate shift (worth investigating), > 0.25 = major shift
    (the feature's learned relationship to the target may no longer hold).
    These thresholds are the standard convention from credit-risk and churn
    model monitoring, not this project's invention.
  - **Kolmogorov-Smirnov test**: a distribution-free test of whether two
    continuous samples come from the same distribution. It gives a p-value
    PSI doesn't, and is more sensitive to shifts concentrated in a small
    number of tail values that decile-bucketed PSI can dilute away.

Both are computed here from scratch (scipy for the KS test, plain pandas/
NumPy for PSI) rather than via a monitoring framework such as Evidently:
Evidently's dependency footprint (a full web framework, telemetry, an NLP
toolkit — none of which this project uses elsewhere) is disproportionate to
what is, mathematically, two well-defined statistics over two dataframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.quality import clean_customer_features
from src.eda import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS, save_figure, set_style
from src.explainability import DEFAULT_RISK_HIGH_CUTOFF, DEFAULT_RISK_LOW_CUTOFF, risk_level_from_probability
from src.models.preprocessing import BOOLEAN_FEATURES, EXCLUDED_WITH_REASON, NUMERIC_FEATURES, split_X_y_tree

PSI_MODERATE_THRESHOLD = 0.10
PSI_MAJOR_THRESHOLD = 0.25

# Every feature the deployed model actually consumes (see
# src/models/preprocessing.py), split by the drift test that fits its type.
NUMERIC_DRIFT_FEATURES = NUMERIC_FEATURES + [c for c in EXCLUDED_WITH_REASON if c != "country_name"]
CATEGORICAL_DRIFT_FEATURES = BOOLEAN_FEATURES + ["country_name"]


def classify_psi(psi: float) -> str:
    """Standard PSI severity bands used in production model monitoring."""
    if psi >= PSI_MAJOR_THRESHOLD:
        return "major"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of `current` against decile edges learned from `reference` only.

    Bin edges come from the reference distribution alone (mirroring the
    project's fit-on-train-only convention elsewhere, e.g.
    `CustomerFeatureEngineer`) — `current` is simply counted into those fixed
    edges. A shifted population is exactly one that lands in different
    proportions across reference's own buckets than reference itself did.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()
    if reference.empty or current.empty:
        raise ValueError("PSI requires at least one non-null value in each sample.")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Reference is (near-)constant — there is no real distribution to
        # bucket, so there is nothing meaningful to call "shifted."
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = pd.cut(reference, bins=edges).value_counts(sort=False) / len(reference)
    cur_pct = pd.cut(current, bins=edges).value_counts(sort=False) / len(current)
    ref_pct, cur_pct = ref_pct.clip(lower=1e-4), cur_pct.clip(lower=1e-4)

    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category proportions instead of numeric bins — same formula.

    The bucket set is every category observed in EITHER sample, so a category
    that is brand-new (or has vanished) in `current` still gets the epsilon
    floor and contributes to PSI, rather than being silently dropped.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()
    categories = pd.Index(reference.unique()).union(current.unique())

    ref_pct = reference.value_counts(normalize=True).reindex(categories, fill_value=0.0).clip(lower=1e-4)
    cur_pct = current.value_counts(normalize=True).reindex(categories, fill_value=0.0).clip(lower=1e-4)

    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def ks_drift_test(reference: pd.Series, current: pd.Series, alpha: float = 0.05) -> dict:
    """Two-sample Kolmogorov-Smirnov test; `drifted` is a plain significance
    call at `alpha` — the p-value itself is also returned for finer judgement.
    """
    statistic, p_value = stats.ks_2samp(pd.Series(reference).dropna(), pd.Series(current).dropna())
    return {"ks_statistic": float(statistic), "p_value": float(p_value), "drifted": bool(p_value < alpha)}


def numeric_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """PSI + KS test for every continuous/ordinal column in `columns`,
    sorted by PSI (most-shifted feature first).
    """
    rows = []
    for col in columns:
        psi = population_stability_index(reference[col], current[col])
        ks = ks_drift_test(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "severity": classify_psi(psi),
                "ks_statistic": round(ks["ks_statistic"], 4),
                "ks_p_value": round(ks["p_value"], 6),
                "ks_drifted": ks["drifted"],
                "reference_mean": round(float(reference[col].mean()), 4),
                "current_mean": round(float(current[col].mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def categorical_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """PSI (over category/boolean proportions) for every discrete column in
    `columns`, sorted by PSI (most-shifted feature first).
    """
    rows = []
    for col in columns:
        psi = categorical_psi(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "severity": classify_psi(psi),
                "n_categories_reference": int(reference[col].nunique()),
                "n_categories_current": int(current[col].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full analysis (shared by scripts/run_drift_monitoring.py and the dashboard's
# monitoring page, so the two can never quietly compute two different
# versions of "the drift result" — the same principle src/serving.py already
# established for predictions).
# ---------------------------------------------------------------------------


@dataclass
class DriftAnalysis:
    """Every number Step 19's monitoring report and dashboard page need."""

    numeric_report: pd.DataFrame
    categorical_report: pd.DataFrame
    reference_proba: pd.Series
    current_proba: pd.Series
    prediction_psi: float
    prediction_ks: dict
    risk_band_table: pd.DataFrame
    major_features: list[str]


def compute_drift_analysis(
    reference: pd.DataFrame, current_raw: pd.DataFrame, engineer, final_model
) -> DriftAnalysis:
    """Run the full reference-vs-current comparison.

    `reference` must already be engineered (e.g. `train.parquet`).
    `current_raw` must be the SQL pipeline's raw export (has the raw columns,
    not yet cleaned or feature-engineered) — this function applies the exact
    same Step 4 cleaning and TRAIN-fitted `engineer` used to build `reference`,
    so the two populations are compared on identical, leakage-safe terms.
    """
    current_cleaned, _ = clean_customer_features(current_raw)
    current = engineer.transform(current_cleaned)

    numeric_report = numeric_drift_report(reference, current, NUMERIC_DRIFT_FEATURES)
    categorical_report = categorical_drift_report(reference, current, CATEGORICAL_DRIFT_FEATURES)
    major_features = pd.concat([numeric_report, categorical_report])
    major_features = major_features.loc[major_features["severity"] == "major", "feature"].tolist()

    X_reference, _ = split_X_y_tree(reference)
    X_current, _ = split_X_y_tree(current)
    reference_proba = pd.Series(final_model.predict_proba(X_reference)[:, 1], index=reference.index)
    current_proba = pd.Series(final_model.predict_proba(X_current)[:, 1], index=current.index)

    prediction_psi = population_stability_index(reference_proba, current_proba)
    prediction_ks = ks_drift_test(reference_proba, current_proba)

    risk_bands_reference = reference_proba.apply(
        lambda p: risk_level_from_probability(p, DEFAULT_RISK_LOW_CUTOFF, DEFAULT_RISK_HIGH_CUTOFF)
    )
    risk_bands_current = current_proba.apply(
        lambda p: risk_level_from_probability(p, DEFAULT_RISK_LOW_CUTOFF, DEFAULT_RISK_HIGH_CUTOFF)
    )
    risk_band_table = (
        pd.DataFrame(
            {
                "reference_pct": risk_bands_reference.value_counts(normalize=True) * 100,
                "current_pct": risk_bands_current.value_counts(normalize=True) * 100,
            }
        )
        .reindex(["Low", "Medium", "High"])
        .round(2)
    )
    risk_band_table.index.name = "risk_band"

    return DriftAnalysis(
        numeric_report=numeric_report,
        categorical_report=categorical_report,
        reference_proba=reference_proba,
        current_proba=current_proba,
        prediction_psi=prediction_psi,
        prediction_ks=prediction_ks,
        risk_band_table=risk_band_table,
        major_features=major_features,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_probability_drift(
    reference_proba: pd.Series, current_proba: pd.Series, name: str = "drift_probability_distribution"
) -> Path:
    """Overlaid churn-probability distributions: reference (training
    population) vs. current (population being monitored) — the fastest
    visual read on prediction drift, backed by the PSI/KS numbers computed
    alongside it.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(
        reference_proba,
        bins=25,
        range=(0, 1),
        density=True,
        alpha=0.55,
        color=SEQUENTIAL_BLUE,
        label=f"Reference (n={len(reference_proba):,})",
    )
    ax.hist(
        current_proba,
        bins=25,
        range=(0, 1),
        density=True,
        alpha=0.55,
        color=STATUS["critical"],
        label=f"Current (n={len(current_proba):,})",
    )
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction drift: churn probability distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_feature_psi_bar(
    drift_report: pd.DataFrame, top_n: int = 15, name: str = "drift_psi_by_feature"
) -> Path:
    """Horizontal bar of the top-N features by PSI, coloured by severity band,
    with the moderate/major threshold lines drawn for direct visual context.
    """
    set_style()
    top = drift_report.sort_values("psi", ascending=False).head(top_n).iloc[::-1]
    severity_colors = {"none": CATEGORICAL[0], "moderate": STATUS["warning"], "major": STATUS["critical"]}
    bar_colors = [severity_colors[s] for s in top["severity"]]

    fig, ax = plt.subplots(figsize=(6.5, max(2.4, 0.35 * len(top))))
    ax.barh(top["feature"], top["psi"], color=bar_colors)
    ax.axvline(PSI_MODERATE_THRESHOLD, color=INK["muted"], linestyle="--", linewidth=1)
    ax.axvline(PSI_MAJOR_THRESHOLD, color=INK["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title(f"Feature drift: top {len(top)} by PSI (dashed = moderate/major thresholds)")
    fig.tight_layout()
    return save_figure(fig, name)
