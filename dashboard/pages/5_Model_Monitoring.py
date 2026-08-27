"""Model Monitoring — Step 19's drift detection: does the customer
population and the deployed model's predictions still look like what the
model was trained on?
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.charts import render_feature_psi_bar, render_probability_drift  # noqa: E402
from dashboard.data import drift_missing_artifacts, get_drift_analysis, missing_artifacts  # noqa: E402
from src.monitoring import PSI_MAJOR_THRESHOLD, PSI_MODERATE_THRESHOLD  # noqa: E402

st.set_page_config(page_title="Model Monitoring", page_icon="📡", layout="wide")
st.title("Model Monitoring")
st.caption(
    "Reference = the actual training population (cutoff 2011-06-09). Current = a REAL snapshot of "
    "the same business 3 months earlier (cutoff 2011-03-09, `run_pipeline.py --cutoff 2011-03-09 "
    "--horizon 91`) — not a synthetic or resampled dataset. See `reports/monitoring_report.md` for "
    "the full write-up, including the stated limitation on label-based comparisons (see below)."
)

missing = missing_artifacts() + drift_missing_artifacts()
if missing:
    st.error("Missing artefacts: " + ", ".join(f"`{m}`" for m in missing))
    st.stop()

result = get_drift_analysis()
combined = pd.concat([result.numeric_report, result.categorical_report], ignore_index=True)
n_major = int((combined["severity"] == "major").sum())
n_moderate = int((combined["severity"] == "moderate").sum())

st.info(
    "**Limitation, stated directly:** the current snapshot's label uses a 91-day horizon (the "
    "model's own is 183 days), so its `is_churned` column is a different target definition. This "
    "page compares INPUT and PREDICTION drift only — never label-based performance drift."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Features with major drift", n_major, help=f"PSI >= {PSI_MAJOR_THRESHOLD}")
c2.metric(
    "Features with moderate drift",
    n_moderate,
    help=f"PSI in [{PSI_MODERATE_THRESHOLD}, {PSI_MAJOR_THRESHOLD})",
)
c3.metric("Prediction PSI", f"{result.prediction_psi:.4f}", help="PSI on the churn-probability distribution.")
c4.metric(
    "Prediction KS test",
    "Drifted" if result.prediction_ks["drifted"] else "Stable",
    help=f"p-value = {result.prediction_ks['p_value']:.4f} (alpha=0.05)",
)

st.divider()
st.subheader("Feature drift")
st.pyplot(render_feature_psi_bar(combined), use_container_width=False)
st.caption(
    "`tenure_days`, `recency_days`, and `recency_score` (a discretised copy of `recency_days`) are "
    "the only features flagged major — all three are cutoff-relative time measures. That drift is "
    "mechanical, not a sign of a broken pipeline: the business's history only starts 2009-12-01, so "
    "a population observed at the earlier cutoff has structurally had less time to accumulate tenure. "
    "Full reasoning in `reports/monitoring_report.md`."
)

t1, t2 = st.tabs(["Numeric / ordinal features", "Categorical / boolean features"])
with t1:
    st.dataframe(result.numeric_report, use_container_width=True, height=400)
with t2:
    st.dataframe(result.categorical_report, use_container_width=True, height=220)

st.divider()
st.subheader("Prediction drift")
st.pyplot(render_probability_drift(result.reference_proba, result.current_proba), use_container_width=False)

st.subheader("Risk-band distribution (% of customers)")
st.dataframe(result.risk_band_table, use_container_width=False)
