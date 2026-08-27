"""Executive Overview — Customer Intelligence & Churn Decision Platform.

Run:
    streamlit run dashboard/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.charts import render_risk_band_bar  # noqa: E402
from dashboard.data import get_customers_with_risk, missing_artifacts  # noqa: E402

st.set_page_config(page_title="Customer Intelligence Platform", page_icon="📊", layout="wide")

st.title("Customer Intelligence & Churn Decision Platform")
st.caption(
    "Online Retail II · cutoff 2011-06-09 · 183-day churn horizon · "
    "every number on this page is computed live from the project's saved models and data."
)

missing = missing_artifacts()
if missing:
    st.error(
        "Required project artefacts are missing, so this dashboard cannot load real data:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nRun the project pipeline first (see README: Steps 6, 9, 10, 12, 13)."
    )
    st.stop()

df = get_customers_with_risk()

st.header("Executive Overview")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Customers", f"{len(df):,}")
col2.metric("Historical churn rate", f"{df['is_churned'].mean():.1%}")
n_high_risk = int((df["risk_band"] == "High").sum())
col3.metric("Predicted high-risk customers", f"{n_high_risk:,}", help="Calibrated churn probability >= 60%.")
value_at_risk = df["retention_priority_score"].sum()
col4.metric(
    "Estimated value at risk",
    f"€{value_at_risk:,.0f}",
    help="Sum of (churn probability × estimated CLV) across all customers "
    "— Step 12's retention priority score.",
)
n_priority = int((df["risk_value_quadrant"] == "High risk / High value").sum())
col5.metric(
    "Retention-priority customers",
    f"{n_priority:,}",
    help="Step 12's 'High risk / High value' quadrant — likely to churn AND worth saving.",
)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Customers by risk band")
    counts = df["risk_band"].value_counts().to_dict()
    st.pyplot(render_risk_band_bar(counts), use_container_width=False)
    st.caption("Low < 30% · Medium 30-60% · High >= 60% predicted churn probability (Step 11's bands).")

with right:
    st.subheader("Risk vs. value quadrant (Step 12)")
    quadrant_counts = df["risk_value_quadrant"].value_counts()
    st.dataframe(
        quadrant_counts.rename("customers").to_frame(),
        use_container_width=True,
    )
    st.caption(
        '"High risk / High value" is the genuine retention priority — see Step 12\'s finding that '
        "ranking by churn probability alone would target a substantially different, lower-value set "
        "of customers (0% list overlap, measured on this data)."
    )

st.divider()
st.markdown(
    "**Use the sidebar** to explore churn analytics, model performance, an individual customer, "
    "or the behavioural segments (Steps 5, 7-11, and 13 respectively)."
)
