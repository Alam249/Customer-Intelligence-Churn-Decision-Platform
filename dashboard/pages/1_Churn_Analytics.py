"""Churn Analytics — the real EDA findings from Step 5, with the actual
figures that produced them (not redrawn here — displaying the same PNGs the
EDA notebook generated keeps this page and the notebook from ever disagreeing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PATHS  # noqa: E402

st.set_page_config(page_title="Churn Analytics", page_icon="📈", layout="wide")
st.title("Churn Analytics")
st.caption("Step 5 EDA findings — figures are the actual PNGs the EDA notebook produced.")

FIG = PATHS.figures


def figure(col, path: Path, caption: str) -> None:
    if path.is_file():
        col.image(str(path), caption=caption, use_container_width=True)
    else:
        col.warning(f"Figure not found: {path.name} — run scripts/run_eda... (Step 5) to generate it.")


st.subheader("Target")
c1, c2 = st.columns([1, 2])
figure(c1, FIG / "target_balance.png", "42.5% churned / 57.5% retained — mild imbalance.")
figure(c2, FIG / "correlation_heatmap.png", "Feature correlation matrix (RFM core + activity).")

st.divider()
st.subheader("Behavioural relationships with churn")
c1, c2 = st.columns(2)
figure(c1, FIG / "binned_monetary_total_churn_rate.png", "Spend vs. churn — clean, near-linear gradient.")
figure(c2, FIG / "binned_tenure_days_churn_rate.png", "Tenure vs. churn — non-monotonic, not linear.")

c1, c2 = st.columns(2)
figure(c1, FIG / "binned_distinct_products_churn_rate.png", "Catalogue breadth vs. churn — protective.")
figure(c2, FIG / "churnrate_country_name.png", "Churn by country — mostly a UK dataset (91.5%).")

st.divider()
st.subheader("Key Data Science Findings")
st.markdown("""
1. **Churn is well-defined and mildly imbalanced** — 42.5% churned / 57.5% retained (1.35:1) at the
   183-day horizon.
2. **Recency is the single strongest univariate signal.** Median 59 days (retained) vs. 196 days
   (churned); churn climbs from 16% to 69% across recency quintiles. Legitimate — computed strictly
   pre-cutoff — not leakage.
3. **Monetary value shows a clean, near-linear gradient with churn**: 69.1% churn in the bottom
   spending quintile down to 11.9% in the top.
4. **Recent activity (last 90 days) separates customers more sharply than lifetime totals do**: 58.3%
   vs. 23.7% churn by presence/absence of a purchase in the trailing quarter.
5. **Product/category breadth is protective.** Churn falls from 63.6% (fewest distinct products) to
   15.6% (most).
6. **Tenure's relationship with churn is non-monotonic, not linear** — highest in the second-shortest
   tenure bin, not the shortest, consistent with new customers not yet having had a fair chance to
   re-order inside the observation window.
7. **Returning an item is associated with LOWER churn** (29.9% vs. 52.7%) — read as an engagement-level
   correlation (a return requires an active relationship), not a causal effect of returns.
8. **`frequency` and `active_days` carry near-duplicate information** (r = 0.961) — relevant for the
   linear baseline (Step 7), less so for tree-based models (Step 8).
9. **Geography is a weak signal.** `is_uk` alone is not significant (p = 0.44); the full country
   breakdown is (p = 0.007), but most non-UK countries have samples too small (<25 customers) to act on.
10. **Every numeric feature tested (10/10) differs significantly between churned and retained
    customers** (Mann-Whitney, p < 0.001 even after Bonferroni correction).
    """)

st.subheader("Modelling Implications")
st.markdown("""
- **Class weighting, not resampling** — the imbalance is mild enough that `class_weight="balanced"` is
  sufficient (used from the Step 7 baseline onward).
- **Log/power-transform monetary and count features for linear models** — Step 7's Logistic Regression
  uses a Yeo-Johnson transform for exactly this reason.
- **`recency_days` is expected to dominate feature importance — not a red flag.** Step 11's SHAP
  analysis confirms this and explains why it isn't leakage.
- **Churn probability and customer value move in opposite directions** — the reason Step 12 combines
  them into a single retention-priority score instead of ranking on churn probability alone.
    """)
