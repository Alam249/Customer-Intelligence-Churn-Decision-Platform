"""Retention Priority Score: combining churn risk with customer value.

Why churn probability alone is not enough
-------------------------------------------
Every EDA and modelling step so far (5, 7, 9, 11) found monetary value and
churn probability move in OPPOSITE directions — high-value customers churn
less. A retention program that ranks purely by churn probability will
therefore systematically over-target low-value customers: they are
mechanically more likely to cross any probability threshold, while the
highest-value customers at genuine risk get diluted into a much larger list.
`compare_targeting_strategies` below measures this directly rather than
asserting it.

Retention Priority Score
--------------------------
    priority_score = churn_probability * CLV

This is deliberately the simplest defensible formula, not an arbitrarily
"improved" one: it is the expected revenue lost if a customer churns and
nothing is done — the same expected-value logic as Step 10's business-cost
framework, applied per customer instead of per threshold. Ranking by this
score is equivalent to ranking by "how much money is genuinely at stake here,"
which is what a retention budget should actually be spent protecting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eda import CATEGORICAL, INK, save_figure, set_style

set_style()

SEGMENT_COLORS = {
    "High risk / High value": CATEGORICAL[7],  # red — the segment that matters most
    "High risk / Low value": CATEGORICAL[3],  # yellow
    "Low risk / High value": CATEGORICAL[0],  # blue
    "Low risk / Low value": CATEGORICAL[2],  # aqua
}


def compute_retention_priority(
    df: pd.DataFrame, churn_col: str = "churn_probability", clv_col: str = "clv"
) -> pd.DataFrame:
    out = df.copy()
    out["retention_priority_score"] = (out[churn_col] * out[clv_col]).round(2)
    return out


def assign_segments(
    df: pd.DataFrame,
    churn_col: str = "churn_probability",
    clv_col: str = "clv",
) -> pd.DataFrame:
    """High/Low risk x High/Low value quadrants, split at each dimension's
    own median — a standard, simple segmentation that adapts to this
    population rather than using arbitrary fixed cutoffs.
    """
    out = df.copy()
    risk_median = out[churn_col].median()
    value_median = out[clv_col].median()

    is_high_risk = out[churn_col] >= risk_median
    is_high_value = out[clv_col] >= value_median

    out["segment"] = "Low risk / Low value"
    out.loc[is_high_risk & is_high_value, "segment"] = "High risk / High value"
    out.loc[is_high_risk & ~is_high_value, "segment"] = "High risk / Low value"
    out.loc[~is_high_risk & is_high_value, "segment"] = "Low risk / High value"

    out.attrs["risk_median"] = risk_median
    out.attrs["value_median"] = value_median
    return out


def segment_summary(
    df: pd.DataFrame, churn_col: str = "churn_probability", clv_col: str = "clv"
) -> pd.DataFrame:
    summary = df.groupby("segment").agg(
        n_customers=("customer_id", "count"),
        avg_churn_probability=(churn_col, "mean"),
        avg_clv=(clv_col, "mean"),
        total_clv_at_risk=(clv_col, lambda s: (s * df.loc[s.index, churn_col]).sum()),
    )
    return summary.round(2).sort_values("total_clv_at_risk", ascending=False)


def compare_targeting_strategies(
    df: pd.DataFrame,
    top_n: int,
    churn_col: str = "churn_probability",
    clv_col: str = "clv",
) -> dict:
    """Concrete evidence for why churn probability alone is insufficient:
    total CLV captured by a top-N list ranked on churn probability alone vs.
    ranked on the combined retention priority score, for the SAME contact
    budget (same list size).
    """
    by_churn_only = df.nlargest(top_n, churn_col)
    by_priority = df.nlargest(top_n, "retention_priority_score")

    return {
        "top_n": top_n,
        "churn_only_avg_clv": round(by_churn_only[clv_col].mean(), 2),
        "churn_only_total_clv_at_risk": round((by_churn_only[clv_col] * by_churn_only[churn_col]).sum(), 2),
        "priority_avg_clv": round(by_priority[clv_col].mean(), 2),
        "priority_total_clv_at_risk": round((by_priority[clv_col] * by_priority[churn_col]).sum(), 2),
        "overlap_pct": round(
            len(set(by_churn_only["customer_id"]) & set(by_priority["customer_id"])) / top_n * 100, 1
        ),
    }


def plot_segment_quadrant(
    df: pd.DataFrame,
    churn_col: str = "churn_probability",
    clv_col: str = "clv",
    name: str = "retention_quadrant",
) -> Path:
    """Every customer plotted by churn probability (x) vs. CLV (y, log scale),
    coloured by segment, with the median split lines that define the quadrants.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for segment, color in SEGMENT_COLORS.items():
        subset = df[df["segment"] == segment]
        ax.scatter(
            subset[churn_col],
            subset[clv_col].clip(lower=1),
            s=10,
            alpha=0.5,
            color=color,
            label=f"{segment} (n={len(subset)})",
        )
    risk_median = df.attrs.get("risk_median", df[churn_col].median())
    value_median = df.attrs.get("value_median", df[clv_col].median())
    ax.axvline(risk_median, color=INK["muted"], linestyle="--", linewidth=1)
    ax.axhline(value_median, color=INK["muted"], linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Estimated CLV (log scale)")
    ax.set_title("Retention segments: risk vs. value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_targeting_comparison(comparison: dict, name: str = "targeting_comparison") -> Path:
    """Same contact-list size, two ranking strategies: total CLV-at-risk
    captured by ranking on churn probability alone vs. the combined priority
    score — the concrete evidence for why churn probability alone falls short.
    """
    labels = ["Ranked by\nchurn probability alone", "Ranked by\nretention priority score"]
    values = [comparison["churn_only_total_clv_at_risk"], comparison["priority_total_clv_at_risk"]]
    colors = [INK["muted"], CATEGORICAL[0]]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"€{v:,.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Total CLV at risk captured (€)")
    ax.set_title(f"Same contact budget (top {comparison['top_n']} customers)")
    fig.tight_layout()
    return save_figure(fig, name)
