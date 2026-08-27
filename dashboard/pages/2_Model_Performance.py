"""Model Performance — live metrics against the real held-out test set, plus
an interactive decision-threshold explorer (Step 10's lesson, made explorable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.charts import render_confusion_matrix  # noqa: E402
from dashboard.data import get_model_comparison, get_test_predictions, missing_artifacts  # noqa: E402
from src.config import PATHS  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402

st.set_page_config(page_title="Model Performance", page_icon="🎯", layout="wide")
st.title("Model Performance")

missing = missing_artifacts()
if missing:
    st.error("Missing artefacts: " + ", ".join(f"`{m}`" for m in missing))
    st.stop()

st.caption(
    "Every metric below is computed right now against `data/processed/test.parquet` using the actual "
    "saved model files — not copied from the Step 7-10 reports."
)

st.subheader("Model comparison (test set)")
comparison = get_model_comparison()
st.dataframe(
    comparison.style.format("{:.4f}").highlight_max(subset=["roc_auc", "pr_auc"], color="#d6eaf8"),
    use_container_width=True,
)
st.caption(
    "Step 8's finding, reproduced live: the untuned Logistic Regression baseline and Random Forest are "
    "competitive with untuned XGBoost — the tuned/calibrated final model (Step 9-10) is the one actually "
    "deployed. See `reports/model_comparison_report.md` for the full discussion, including why the "
    '"most complex model" did not automatically win here.'
)

FIG = PATHS.figures
c1, c2 = st.columns(2)
if (FIG / "roc_comparison_tuning.png").is_file():
    c1.image(
        str(FIG / "roc_comparison_tuning.png"),
        caption="Tuned vs. untuned XGBoost (Step 9)",
        use_container_width=True,
    )
if (FIG / "calibration_curve.png").is_file():
    c2.image(
        str(FIG / "calibration_curve.png"),
        caption="Calibration: raw vs. isotonic (Step 10)",
        use_container_width=True,
    )

st.divider()
st.subheader("Decision threshold explorer")
st.markdown(
    "The final model doesn't use a threshold of 0.50 by default (Step 10). Move the slider to see how "
    "precision, recall, and the confusion matrix trade off — computed live on the real test set."
)

preds = get_test_predictions()
threshold = st.slider("Decision threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

y_true = preds["is_churned"]
y_proba = preds["churn_probability"]
y_pred = (y_proba >= threshold).astype(int)
metrics = compute_classification_metrics(y_true, y_pred, y_proba)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{metrics['precision']:.3f}")
m2.metric("Recall", f"{metrics['recall']:.3f}")
m3.metric("F1", f"{metrics['f1']:.3f}")
m4.metric("% flagged as churn risk", f"{y_pred.mean():.1%}")

c1, c2 = st.columns([1, 1])
c1.pyplot(render_confusion_matrix(y_true, y_pred), use_container_width=False)
with c2:
    st.markdown(f"""
**At threshold {threshold:.2f}** (test set, {len(preds):,} customers):

- ROC-AUC and PR-AUC don't change with threshold (they're threshold-independent, {metrics['roc_auc']:.3f}
  and {metrics['pr_auc']:.3f} respectively) — only precision/recall/F1 do.
- Step 10's business-cost framework found the *cost-optimal* threshold for its stated (hypothetical)
  contact-cost assumptions was much lower than 0.50 — see
  `reports/calibration_threshold_report.md` for the full reasoning, including the caveat that the
  cost-optimal number alone isn't automatically an *operationally* usable contact list.
        """)

st.caption(
    "Note: `threshold_curves.png` (Step 10) shows the full precision/recall curve across every "
    "threshold from a single static image; this slider recomputes the same real test-set predictions "
    "at whichever single threshold you choose."
)
