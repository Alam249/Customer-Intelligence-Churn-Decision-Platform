"""Uplift & Targeting — Step 20's uplift modeling: given a (simulated)
retention campaign, which customers would actually respond to being
contacted, as opposed to just being high risk?
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.charts import render_qini_curves, render_uplift_by_decile  # noqa: E402
from dashboard.data import get_uplift_analysis, missing_artifacts  # noqa: E402
from src.uplift import TOP_K_FRACTION  # noqa: E402

st.set_page_config(page_title="Uplift & Targeting", page_icon="🎯", layout="wide")
st.title("Uplift & Targeting")

st.warning(
    "**Every treatment-effect number on this page is SIMULATED, not measured.** Online Retail II has "
    "no real retention campaign — no customer here was ever randomly offered a discount or a retention "
    "email. The simulation is built on REAL customer covariates and the REAL Step 10 model's baseline "
    "churn probability; only the treatment assignment and effect mechanism are synthetic. See "
    "`reports/uplift_modeling_report.md` and `src/uplift.py`'s module docstring for the full design."
)
st.caption(
    "Step 12 ranked customers by `churn_probability x CLV` — how likely someone is to leave, times how "
    "much they're worth. That says nothing about whether contacting them would change their behaviour. "
    "This page asks that different question: given a randomised experiment, who genuinely responds?"
)

missing = missing_artifacts()
if missing:
    st.error("Missing artefacts: " + ", ".join(f"`{m}`" for m in missing))
    st.stop()

result = get_uplift_analysis()
best_real_model = result.auuc_table.loc[result.auuc_table["model"] != "Oracle (true uplift)", "model"].iloc[0]
best_real_auuc = result.auuc_table.loc[result.auuc_table["model"] == best_real_model, "auuc"].iloc[0]
oracle_auuc = result.auuc_table.loc[result.auuc_table["model"] == "Oracle (true uplift)", "auuc"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best real model", best_real_model, help="Ranked by AUUC among S-/T-/X-learner and naive risk.")
c2.metric(f"{best_real_model} AUUC", f"{best_real_auuc:.2f}", help="Higher = better than random targeting.")
c3.metric("Oracle AUUC (ceiling)", f"{oracle_auuc:.2f}", help="Best possible — only knowable in simulation.")
c4.metric(
    f"Overlap with Step 12 top {int(TOP_K_FRACTION * 100)}%",
    f"{result.overlap_pct:.1f}%",
    help="X-learner's top targets vs. retention_priority_score's top targets — same customers?",
)

st.divider()
st.subheader("Model ranking (AUUC)")
st.dataframe(result.auuc_table, use_container_width=False, hide_index=True)
st.pyplot(render_qini_curves(result.qini_results), use_container_width=False)
st.caption(
    "AUUC is a cumulative statistic with real sampling noise even under no true effect (confirmed "
    "directly while building this step) — read alongside the Qini curve shape, not instead of it."
)

st.divider()
st.subheader("Why S-learner underperforms despite decent ground-truth correlation")
st.dataframe(result.spread_table, use_container_width=False, hide_index=True)
st.caption(
    "S-learner's predicted uplift has a far smaller spread than T-/X-learner's or the true effect's "
    "own spread — a textbook symptom of treatment being just one more feature among many, so a single "
    "flexible model has little incentive to actually use it."
)

st.divider()
st.subheader("Validation against ground truth (only possible because this is a simulation)")
gt_table = result.ground_truth_correlation
precision_table = result.precision_at_k
st.dataframe(
    {
        "Score": list(gt_table.keys()),
        "Correlation with true uplift": [f"{gt_table[k]:.3f}" for k in gt_table],
        f"Precision@{int(TOP_K_FRACTION * 100)}% (base rate {result.base_rate:.1%})": [
            f"{precision_table[k]:.1%}" for k in gt_table
        ],
    },
    use_container_width=False,
    hide_index=True,
)
st.pyplot(render_uplift_by_decile(result.decile_table), use_container_width=False)
st.caption(
    "The overall downward trend from D9 to D0 is real; individual bars are not perfectly monotonic — "
    "expected at this population size (~430 customers per decile), not a bug. See the full report for "
    "the same check on the simulation's own ground-truth score."
)
