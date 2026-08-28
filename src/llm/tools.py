"""Grounding tools for the LLM analyst layer (Step 21).

Why tools, not free-text generation
------------------------------------
An LLM asked "what's this customer's churn risk" has no real way to know the
answer — it can only generate something plausible, which for a churn
platform is exactly the failure mode that matters most: a confident, wrong
number. Every function here wraps an ALREADY-BUILT, ALREADY-TESTED piece of
this project (Step 10's calibrated model, Step 11's SHAP explainer, Step
12's CLV/priority scores, Step 13's segments, Step 19's drift analysis, Step
20's SIMULATED uplift analysis) and returns real, live-computed numbers —
never a hardcoded or remembered figure. `src/llm/agent.py`'s system prompt
instructs the model to call a tool for every factual claim; this module is
what makes that instruction enforceable rather than aspirational.

Every function returns a plain, JSON-serialisable dict so it can cross an
LLM tool-call boundary unchanged — an "error" key rather than a raised
exception for bad input, since a raised exception would otherwise crash the
whole conversation instead of giving the model something it can react to
(e.g. re-asking the user for a valid customer ID).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import joblib
import pandas as pd

from src.config import PATHS
from src.evaluation.metrics import compute_classification_metrics
from src.explainability import (
    DEFAULT_RISK_HIGH_CUTOFF,
    DEFAULT_RISK_LOW_CUTOFF,
    explain_customer,
    risk_level_from_probability,
)
from src.models.preprocessing import split_X_y_tree
from src.monitoring import compute_drift_analysis
from src.serving import ServingContext, load_serving_context
from src.uplift import compute_uplift_analysis

RANK_BY_OPTIONS = ("retention_priority_score", "churn_probability", "clv")


@lru_cache(maxsize=1)
def _context() -> ServingContext:
    """Loaded once per process — models and the customer table never reload
    on every question, the same discipline `api/state.py` and the dashboard
    already apply.
    """
    return load_serving_context()


def get_customer_summary(customer_id: int) -> dict[str, Any]:
    """Real churn probability, risk band, CLV, retention priority, segment,
    and country for one customer — the same numbers `/predict` and the
    Customer Explorer dashboard page show.
    """
    ctx = _context()
    try:
        row = ctx.get_customer_row(int(customer_id))
    except KeyError:
        return {
            "error": f"Customer {customer_id} not found among the {len(ctx.customers):,} customers "
            "this project serves (Online Retail II IDs roughly 12346-18287)."
        }
    r = row.iloc[0]
    return {
        "customer_id": int(customer_id),
        "churn_probability": round(float(r["churn_probability"]), 4),
        "risk_level": risk_level_from_probability(
            r["churn_probability"], DEFAULT_RISK_LOW_CUTOFF, DEFAULT_RISK_HIGH_CUTOFF
        ),
        "clv_eur": None if pd.isna(r["clv"]) else round(float(r["clv"]), 2),
        "retention_priority_score": (
            None if pd.isna(r["retention_priority_score"]) else round(float(r["retention_priority_score"]), 2)
        ),
        "risk_value_quadrant": None if pd.isna(r["risk_value_quadrant"]) else str(r["risk_value_quadrant"]),
        "segment_name": None if pd.isna(r["segment_name"]) else str(r["segment_name"]),
        "country": str(r["country_name"]),
    }


def explain_customer_churn(customer_id: int) -> dict[str, Any]:
    """Real SHAP-based explanation (Step 11): the top factors pushing this
    customer's churn probability up or down, plus a plain-English narrative.
    """
    ctx = _context()
    try:
        result = explain_customer(
            int(customer_id),
            ctx.customers,
            ctx.tuned_pipeline,
            ctx.final_model,
            save_plot=False,
            explainer=ctx.explainer,
        )
    except KeyError:
        return {"error": f"Customer {customer_id} not found among the served population."}
    return {
        "customer_id": int(customer_id),
        "churn_probability": round(result["churn_probability"], 4),
        "risk_level": result["risk_level"],
        "top_risk_factors": result["top_risk_factors"],
        "top_protective_factors": result["top_protective_factors"],
        "narrative": result["narrative"],
    }


def get_population_overview() -> dict[str, Any]:
    """Real aggregate statistics across the full served customer population
    — churn rate, CLV totals, and counts by risk/value quadrant and segment.
    """
    df = _context().customers
    return {
        "n_customers": int(len(df)),
        "overall_churn_rate": round(float(df["is_churned"].mean()), 4),
        "mean_churn_probability": round(float(df["churn_probability"].mean()), 4),
        "mean_clv_eur": round(float(df["clv"].mean()), 2),
        "total_clv_eur": round(float(df["clv"].sum()), 2),
        "customers_by_risk_value_quadrant": {
            str(k): int(v) for k, v in df["risk_value_quadrant"].value_counts(dropna=True).items()
        },
        "customers_by_segment": {
            str(k): int(v) for k, v in df["segment_name"].value_counts(dropna=True).items()
        },
    }


def get_top_customers(n: int = 10, rank_by: str = "retention_priority_score") -> dict[str, Any]:
    """Real top-N customers ranked by a chosen real metric — for building a
    concrete, actionable retention contact list.
    """
    if rank_by not in RANK_BY_OPTIONS:
        return {"error": f"rank_by must be one of {list(RANK_BY_OPTIONS)}, got '{rank_by}'."}
    df = _context().customers
    n_capped = max(1, min(int(n), 50))
    top = df.nlargest(n_capped, rank_by)[
        ["customer_id", "churn_probability", "clv", "retention_priority_score", "segment_name"]
    ].round(2)
    return {"rank_by": rank_by, "n": n_capped, "customers": top.to_dict(orient="records")}


def get_model_performance() -> dict[str, Any]:
    """Real Step 10 test-set classification metrics, recomputed live on every
    call against the actual held-out test split — never a stale figure
    copied from a report.
    """
    ctx = _context()
    test_df = pd.read_parquet(PATHS.data_processed / "test.parquet")
    X_test, y_test = split_X_y_tree(test_df)
    proba = ctx.final_model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = compute_classification_metrics(y_test, pred, proba)
    return {
        "evaluated_on": f"{len(test_df):,} held-out test customers (Step 6's stratified split)",
        "decision_threshold": 0.5,
        **{k: round(float(v), 4) for k, v in metrics.items()},
    }


def get_segment_profile(segment_name: str | None = None) -> dict[str, Any]:
    """Real Step 13 K-Means segment profile for one named segment, or the
    list of available segment names if none or an unknown one is given.
    """
    df = _context().customers
    available = sorted(df["segment_name"].dropna().unique().tolist())
    if segment_name not in available:
        return {
            "error": f"Unknown segment '{segment_name}'." if segment_name else "No segment name given.",
            "available_segments": available,
        }
    subset = df[df["segment_name"] == segment_name]
    return {
        "segment_name": segment_name,
        "n_customers": int(len(subset)),
        "churn_rate": round(float(subset["is_churned"].mean()), 4),
        "median_clv_eur": round(float(subset["clv"].median()), 2),
        "median_recency_days": round(float(subset["recency_days"].median()), 1),
        "median_frequency": round(float(subset["frequency"].median()), 1),
        "median_tenure_days": round(float(subset["tenure_days"].median()), 1),
    }


@lru_cache(maxsize=1)
def _cached_drift_analysis():
    reference = pd.read_parquet(PATHS.data_processed / "train.parquet")
    current_raw = pd.read_parquet(PATHS.data_processed / "customer_features_2011-03-09_h91.parquet").drop(
        columns=["cutoff_date"]
    )
    engineer = joblib.load(PATHS.models / "feature_engineer.joblib")
    return compute_drift_analysis(reference, current_raw, engineer, _context().final_model)


def get_drift_status() -> dict[str, Any]:
    """Real Step 19 monitoring result: does the customer population the
    model would score today still look like the population it was trained
    on? Cached after the first call (the underlying analysis is cheap but
    not free).
    """
    result = _cached_drift_analysis()
    combined = pd.concat([result.numeric_report, result.categorical_report])
    major = combined.loc[combined["severity"] == "major", "feature"].tolist()
    return {
        "n_features_major_drift": len(major),
        "features_with_major_drift": major,
        "prediction_drift_psi": round(result.prediction_psi, 4),
        "prediction_ks_p_value": round(result.prediction_ks["p_value"], 6),
        "prediction_drifted": result.prediction_ks["drifted"],
        "note": "Compares the training population (June 2011 cutoff) against a REAL customer "
        "snapshot from March 2011 — see reports/monitoring_report.md for the full write-up.",
    }


@lru_cache(maxsize=1)
def _cached_uplift_analysis():
    return compute_uplift_analysis(_context().customers)


def get_uplift_summary() -> dict[str, Any]:
    """SIMULATED Step 20 uplift-modeling result. Online Retail II has no real
    retention campaign, so every number here comes from a designed synthetic
    experiment on top of real customer covariates — NOT a real measured
    business outcome. The first call takes roughly 30 seconds (cross-fitting
    3 models over the full population); cached after that.
    """
    result = _cached_uplift_analysis()
    return {
        "SIMULATED": True,
        "note": "Online Retail II has no real retention campaign — this is a SIMULATED experiment "
        "for methodology demonstration only, built on real customer covariates. Never present "
        "these numbers as a real measured business outcome.",
        "auuc_ranking": result.auuc_table.to_dict(orient="records"),
        "overlap_with_retention_priority_ranking_pct": round(result.overlap_pct, 1),
    }


@dataclass
class Tool:
    """One callable tool: its name/description/JSON-schema for the LLM, and
    the real Python function that actually executes it.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., dict[str, Any]]


TOOLS: list[Tool] = [
    Tool(
        name="get_customer_summary",
        description="Get a real customer's churn probability, risk band, CLV, retention priority "
        "score, segment, and country.",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "integer", "description": "The customer's ID."}},
            "required": ["customer_id"],
        },
        function=get_customer_summary,
    ),
    Tool(
        name="explain_customer_churn",
        description="Get a SHAP-based explanation of what is driving one customer's churn "
        "prediction — the top risk-increasing and risk-decreasing factors.",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "integer", "description": "The customer's ID."}},
            "required": ["customer_id"],
        },
        function=explain_customer_churn,
    ),
    Tool(
        name="get_population_overview",
        description="Get aggregate statistics across the whole customer population: overall churn "
        "rate, mean/total CLV, and customer counts by risk/value quadrant and by segment.",
        parameters={"type": "object", "properties": {}},
        function=get_population_overview,
    ),
    Tool(
        name="get_top_customers",
        description="Get the top N customers ranked by a chosen metric, for building a concrete "
        "contact list.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many customers to return (max 50)."},
                "rank_by": {
                    "type": "string",
                    "enum": list(RANK_BY_OPTIONS),
                    "description": "Which real metric to rank by.",
                },
            },
            "required": ["n", "rank_by"],
        },
        function=get_top_customers,
    ),
    Tool(
        name="get_model_performance",
        description="Get the deployed churn model's real classification metrics (accuracy, "
        "precision, recall, F1, ROC-AUC, PR-AUC), recomputed live on the actual held-out test set.",
        parameters={"type": "object", "properties": {}},
        function=get_model_performance,
    ),
    Tool(
        name="get_segment_profile",
        description="Get the profile (churn rate, CLV, RFM medians) of one named K-Means customer "
        "segment. Call with no segment_name first to see the list of available segments.",
        parameters={
            "type": "object",
            "properties": {"segment_name": {"type": "string", "description": "The segment's name."}},
        },
        function=get_segment_profile,
    ),
    Tool(
        name="get_drift_status",
        description="Check whether the customer population and the model's predictions have "
        "drifted from what the model was trained on (real Step 19 monitoring analysis).",
        parameters={"type": "object", "properties": {}},
        function=get_drift_status,
    ),
    Tool(
        name="get_uplift_summary",
        description="Get the SIMULATED uplift-modeling analysis: which targeting strategy would "
        "best identify customers who genuinely respond to a retention campaign. ALWAYS labeled "
        "simulated in the result — relay that caveat to the user. Slow on first call (~30s).",
        parameters={"type": "object", "properties": {}},
        function=get_uplift_summary,
    ),
]
