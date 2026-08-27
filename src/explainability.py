"""SHAP-based explainability for the churn model.

Which model SHAP explains, and why
-----------------------------------
The project's final model (`models/final_churn_model.joblib`, Step 10) is a
``CalibratedClassifierCV`` — internally an ENSEMBLE of 5 cloned pipelines, one
per calibration fold, each wrapping its own recalibration curve. SHAP's
``TreeExplainer`` needs a single tree ensemble to walk, not a calibration
wrapper around several of them, so it cannot be pointed at that object.

The split used throughout this module:
  - **SHAP attribution** (which features drive the decision, and by how much)
    comes from the pre-calibration tuned XGBoost pipeline
    (`models/xgboost_tuned.joblib`). Calibration (Step 10) rescales the final
    probability for reliability; it does not change which features the
    underlying trees split on or their relative influence.
  - **The displayed probability** for any customer comes from the calibrated
    final model, so what a stakeholder sees here matches what the business
    threshold (Step 10) and the API (Step 14) will show.

What SHAP values mean — and do not mean
-----------------------------------------
A SHAP value is the feature's contribution to THIS model's THIS prediction,
relative to the average prediction over the background data — a precise,
model-specific accounting number, not a statement about the real world. In
particular:
  - SHAP values are computed in log-odds (margin) space here, the
    ``TreeExplainer`` default. A positive value means the feature pushed the
    model's score toward churn; a negative value pushed it toward retention.
    The magnitude is not directly "N percentage points of probability."
  - **A high-impact feature is not a cause of churn.** SHAP explains the
    MODEL, not the customer's actual decision-making — it cannot separate a
    genuine causal driver from a feature that merely correlates with one
    (e.g. `recency_days` is mechanically close to how churn is labelled;
    Step 5/9 already discuss this). Causal claims need the experimental
    design in Step 20, not an explainability method applied to an
    observational model.
  - Explanations are relative to the background sample used, and can shift if
    that background changes — they are not an absolute, context-free property
    of the customer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.eda import CHURN_COLORS, INK, SEQUENTIAL_BLUE, save_figure, set_style
from src.models.preprocessing import get_tree_output_feature_names, split_X_y_tree

set_style()

# Single source of truth for the risk bands — imported by api/routers/predict.py
# rather than restated as separate literals, so the API and this module's own
# batch explain_customer() can never quietly drift apart on what "High" means.
DEFAULT_RISK_LOW_CUTOFF = 0.30
DEFAULT_RISK_HIGH_CUTOFF = 0.60


def transform_for_shap(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline's fitted preprocessor and return a labelled DataFrame
    — SHAP needs the exact numeric matrix the trees were trained on, with
    readable column names for every plot to be interpretable.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    feature_names = get_tree_output_feature_names(preprocessor)
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def build_explainer(pipeline) -> shap.TreeExplainer:
    """TreeExplainer on the pipeline's underlying XGBoost model."""
    return shap.TreeExplainer(pipeline.named_steps["model"])


def compute_shap_explanation(
    pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None
) -> tuple[shap.Explanation, pd.DataFrame]:
    """SHAP explanation object plus the transformed (labelled) feature matrix
    it was computed against.

    Pass a pre-built ``explainer`` (e.g. one cached once at API startup —
    see api/state.py) to avoid reconstructing a `TreeExplainer` on every call;
    if omitted, one is built fresh from ``pipeline`` as before.
    """
    X_transformed = transform_for_shap(pipeline, X)
    explainer = explainer or build_explainer(pipeline)
    explanation = explainer(X_transformed)
    return explanation, X_transformed


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------


def plot_shap_summary(explanation: shap.Explanation, name: str = "shap_summary") -> Path:
    """Beeswarm plot: every test customer, every feature, coloured by that
    customer's OWN value for the feature (SHAP's standard convention — kept
    as-is here rather than remapped to the project palette, since red=high
    feature value / blue=low feature value is a widely recognised SHAP
    convention independent of any single project's house style).
    """
    fig = plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(explanation, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    return save_figure(fig, name)


def plot_shap_bar_importance(explanation: shap.Explanation, name: str = "shap_importance") -> Path:
    """Global feature importance: mean(|SHAP value|) per feature.

    Uses the project's own bar-chart style (not SHAP's default bar plot) so
    this sits visually alongside the Step 7 coefficient chart and Step 8
    feature-importance charts as the same kind of comparison.
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    table = pd.DataFrame({"feature": explanation.feature_names, "importance": mean_abs})
    table = table.sort_values("importance", ascending=False).head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(table))))
    ax.barh(table["feature"], table["importance"], color=SEQUENTIAL_BLUE)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    return save_figure(fig, name)


def plot_shap_dependence(explanation: shap.Explanation, feature: str, name: str | None = None) -> Path:
    """Dependence scatter: one feature's value (x) vs. its SHAP value (y) for
    every test customer, coloured by SHAP's automatically-chosen best
    interaction partner.
    """
    fig = plt.figure(figsize=(6, 4.5))
    shap.plots.scatter(explanation[:, feature], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return save_figure(fig, name or f"shap_dependence_{feature}")


# ---------------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------------


def risk_level_from_probability(probability: float, low_cutoff: float, high_cutoff: float) -> str:
    if probability >= high_cutoff:
        return "High"
    if probability >= low_cutoff:
        return "Medium"
    return "Low"


def plot_local_explanation(
    factors: pd.DataFrame, customer_id: int, probability: float, name: str | None = None
) -> Path:
    """Horizontal bar chart of one customer's top SHAP contributors.

    Uses the project's fixed churn-outcome colour convention (red = pushes
    toward churn, blue = pushes toward retention) — the same colours as every
    churn-outcome chart since Step 5, so a reader doesn't have to learn a new
    convention for this one plot.
    """
    ordered = factors.reindex(factors["shap_value"].abs().sort_values().index)
    colors = [CHURN_COLORS[True] if v > 0 else CHURN_COLORS[False] for v in ordered["shap_value"]]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(ordered))))
    ax.barh(ordered["label"], ordered["shap_value"], color=colors)
    ax.axvline(0, color=INK["muted"], linewidth=1)
    ax.set_xlabel("SHAP value (log-odds)")
    ax.set_title(f"Customer {customer_id} — churn probability {probability:.1%}")
    fig.tight_layout()
    return save_figure(fig, name or f"shap_local_{customer_id}")


def explain_customer(
    customer_id: int,
    df: pd.DataFrame,
    tuned_pipeline,
    final_model,
    top_n: int = 5,
    risk_low_cutoff: float = DEFAULT_RISK_LOW_CUTOFF,
    risk_high_cutoff: float = DEFAULT_RISK_HIGH_CUTOFF,
    save_plot: bool = True,
    explainer: shap.TreeExplainer | None = None,
) -> dict[str, Any]:
    """Explain one customer's churn prediction — the reusable function this
    step is built around. Understandable to a Data Scientist (raw SHAP values,
    feature names) and a business stakeholder (the narrative string).

    Parameters
    ----------
    df : the feature table to look the customer up in (typically the test
        split, so the explanation is for a genuinely held-out customer).
    tuned_pipeline : the pre-calibration Step 9 XGBoost pipeline (SHAP source).
    final_model : the calibrated Step 10 model (probability source).
    explainer : an optional pre-built `shap.TreeExplainer` (see
        `compute_shap_explanation`) — pass one built once at startup to avoid
        rebuilding it on every call, e.g. from a request handler.

    Raises
    ------
    KeyError if the customer_id is not present in ``df``.
    """
    if customer_id not in df["customer_id"].values:
        raise KeyError(
            f"customer_id {customer_id} not found in the supplied table "
            f"({len(df)} customers, e.g. {df['customer_id'].iloc[0]}...)."
        )

    row = df.loc[df["customer_id"] == customer_id]
    X_row, _ = split_X_y_tree(row)

    probability = float(final_model.predict_proba(X_row)[:, 1][0])
    risk = risk_level_from_probability(probability, risk_low_cutoff, risk_high_cutoff)

    explanation, X_transformed = compute_shap_explanation(tuned_pipeline, X_row, explainer=explainer)
    shap_values = explanation.values[0]
    feature_values = X_transformed.iloc[0]

    factors = pd.DataFrame(
        {
            "label": [
                f"{f} = {v:.3g}" for f, v in zip(explanation.feature_names, feature_values, strict=True)
            ],
            "feature": explanation.feature_names,
            "value": feature_values.values,
            "shap_value": shap_values,
        }
    ).sort_values("shap_value", ascending=False)

    top_risk = factors[factors["shap_value"] > 0].head(top_n)
    top_protective = factors[factors["shap_value"] < 0].sort_values("shap_value").head(top_n)

    narrative_risk = "; ".join(f"{r.label} (+{r.shap_value:.2f})" for r in top_risk.itertuples())
    narrative_protective = "; ".join(f"{r.label} ({r.shap_value:.2f})" for r in top_protective.itertuples())
    narrative = (
        f"Customer {customer_id}: {probability:.1%} predicted churn probability ({risk} risk). "
        f"Top factors increasing risk: {narrative_risk or 'none'}. "
        f"Top factors reducing risk: {narrative_protective or 'none'}."
    )

    plot_path = None
    if save_plot:
        display_factors = pd.concat([top_risk, top_protective])
        plot_path = plot_local_explanation(display_factors, customer_id, probability)

    return {
        "customer_id": customer_id,
        "churn_probability": probability,
        "risk_level": risk,
        "top_risk_factors": top_risk[["feature", "value", "shap_value"]].to_dict("records"),
        "top_protective_factors": top_protective[["feature", "value", "shap_value"]].to_dict("records"),
        "narrative": narrative,
        "plot_path": str(plot_path) if plot_path else None,
    }
