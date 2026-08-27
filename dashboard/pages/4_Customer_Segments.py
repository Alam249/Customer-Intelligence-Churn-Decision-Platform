"""Customer Segments — Step 13's K-Means segmentation: cluster profiles,
churn/value characteristics, and an explorable customer list per segment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.data import get_customers_with_risk, missing_artifacts  # noqa: E402
from src.config import PATHS  # noqa: E402

st.set_page_config(page_title="Customer Segments", page_icon="🧩", layout="wide")
st.title("Customer Segments")
st.caption(
    "K-Means, K=4 (chosen over the silhouette-best K=3 for a materially more actionable split — see "
    "`reports/segmentation_report.md`). Table below is computed live from the real segmented population; "
    "figures are the actual PNGs Step 13 produced."
)

missing = missing_artifacts()
if missing:
    st.error("Missing artefacts: " + ", ".join(f"`{m}`" for m in missing))
    st.stop()

df = get_customers_with_risk()

st.subheader("Segment profiles")
summary = (
    df.groupby("segment_name")
    .agg(
        customers=("customer_id", "count"),
        churn_rate=("is_churned", "mean"),
        median_clv=("clv", "median"),
        median_recency_days=("recency_days", "median"),
        median_frequency=("frequency", "median"),
        median_tenure_days=("tenure_days", "median"),
    )
    .sort_values("churn_rate")
    .round(3)
)
st.dataframe(
    summary.style.format({"churn_rate": "{:.1%}", "median_clv": "€{:,.0f}"}),
    use_container_width=True,
)

FIG = PATHS.figures
c1, c2 = st.columns(2)
if (FIG / "cluster_profile_heatmap.png").is_file():
    c1.image(
        str(FIG / "cluster_profile_heatmap.png"),
        caption="Standardised cluster profiles",
        use_container_width=True,
    )
if (FIG / "cluster_churn_value.png").is_file():
    c2.image(
        str(FIG / "cluster_churn_value.png"),
        caption="Churn rate and value by cluster",
        use_container_width=True,
    )

st.divider()
st.subheader("Does segmentation add value beyond the supervised model?")
st.markdown("""
Cross-tabulated against the Step 12 risk/value quadrant: **ARI = 0.32** (low-to-moderate) — the
clusters are *not* simply re-deriving that simpler 2×2 split. "Declining" and "New" customers, in
particular, spread across three of the four risk/value quadrants each, showing segmentation surfaces
tenure/engagement structure the risk × value view alone collapses away. Full evidence, including the
crosstab, in `reports/segmentation_report.md`.
    """)

st.divider()
st.subheader("Explore customers within a segment")
segment_choice = st.selectbox("Segment", options=sorted(df["segment_name"].unique()))
filtered = df.loc[
    df["segment_name"] == segment_choice,
    [
        "customer_id",
        "churn_probability",
        "clv",
        "retention_priority_score",
        "recency_days",
        "frequency",
        "monetary_total",
        "tenure_days",
        "country_name",
    ],
].sort_values("retention_priority_score", ascending=False)

st.dataframe(filtered, use_container_width=True, height=350)
st.caption(f'{len(filtered):,} customers in "{segment_choice}", sorted by retention priority score.')
