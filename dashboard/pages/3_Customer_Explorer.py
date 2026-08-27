"""Customer Explorer — pick any customer and see their prediction, value,
and SHAP explanation, computed live via the same `explain_customer()`
function used by Step 11's batch report and the Step 14 API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.charts import render_local_shap_bar  # noqa: E402
from dashboard.data import get_context, get_customers_with_risk, missing_artifacts, risk_band  # noqa: E402
from src.explainability import explain_customer  # noqa: E402

st.set_page_config(page_title="Customer Explorer", page_icon="🔍", layout="wide")
st.title("Customer Explorer")

missing = missing_artifacts()
if missing:
    st.error("Missing artefacts: " + ", ".join(f"`{m}`" for m in missing))
    st.stop()

df = get_customers_with_risk()
customer_ids = sorted(df["customer_id"].tolist())
default_index = customer_ids.index(12346) if 12346 in customer_ids else 0

customer_id = st.selectbox(
    "Customer ID",
    options=customer_ids,
    index=default_index,
    help=f"{len(customer_ids):,} customers available "
    "(this project's historical Online Retail II population).",
)

row = df.loc[df["customer_id"] == customer_id].iloc[0]

st.subheader(f"Customer {customer_id}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Churn probability", f"{row['churn_probability']:.1%}")
c2.metric("Risk band", risk_band(row["churn_probability"]))
c3.metric("Estimated CLV (6mo)", f"€{row['clv']:,.0f}")
c4.metric("Retention priority", f"€{row['retention_priority_score']:,.0f}")
c5.metric("Segment", row["segment_name"])

st.divider()
st.subheader("Customer attributes")
attr_cols = st.columns(4)
attrs = [
    ("Recency (days)", f"{row['recency_days']:.0f}"),
    ("Frequency (orders)", f"{row['frequency']:.0f}"),
    ("Lifetime spend", f"€{row['monetary_total']:,.0f}"),
    ("Tenure (days)", f"{row['tenure_days']:.0f}"),
    ("Distinct products", f"{row['distinct_products']:.0f}"),
    ("Country", row["country_name"]),
    ("Risk × value quadrant", row["risk_value_quadrant"]),
    ("Actually churned (historical)", "Yes" if row["is_churned"] else "No"),
]
for i, (label, value) in enumerate(attrs):
    attr_cols[i % 4].metric(label, value)

st.divider()
st.subheader("Why this prediction — SHAP explanation")
st.caption(
    "SHAP attribution runs on the pre-calibration tuned XGBoost model (Step 9); the probability shown "
    "above is from the calibrated final model (Step 10) — the same split established in Step 11, and "
    "the exact same `explain_customer()` function used there, not reimplemented for this page."
)

context = get_context()
with st.spinner("Computing SHAP explanation..."):
    result = explain_customer(
        customer_id,
        context.customers,
        context.tuned_pipeline,
        context.final_model,
        explainer=context.explainer,
        save_plot=False,
    )

st.markdown(f"> {result['narrative']}")

if result["top_risk_factors"] or result["top_protective_factors"]:
    fig = render_local_shap_bar(
        result["top_risk_factors"],
        result["top_protective_factors"],
        customer_id,
        result["churn_probability"],
    )
    st.pyplot(fig, use_container_width=False)

st.info(
    "**What this does and does not mean**: a SHAP value is this feature's contribution to THIS model's "
    "THIS prediction, in log-odds — not a percentage-point probability contribution, and not a causal "
    "claim about why the customer will actually churn. See `reports/shap_explainability_report.md` "
    "(Step 11) for the full explanation."
)
