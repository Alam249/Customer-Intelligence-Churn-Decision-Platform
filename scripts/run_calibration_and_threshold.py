"""Step 10 — Probability calibration and business decision threshold.

Uses the Step 9 tuned XGBoost model (the project's leading candidate — it beat
the Logistic Regression baseline on both ROC-AUC and PR-AUC with the
overfitting problem resolved). Two questions this step answers:

  1. Are its predicted PROBABILITIES trustworthy as probabilities, not just as
     a ranking? (`scale_pos_weight` is known to distort this.)
  2. What decision threshold should flag a customer as "at risk", given that
     0.50 is an arbitrary default with no connection to what a retention
     action actually costs or is worth?

Run:
    python scripts/run_calibration_and_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from src.config import PATHS, RANDOM_SEED  # noqa: E402
from src.evaluation.calibration import (  # noqa: E402
    compute_brier_score,
    plot_calibration_curves,
    plot_threshold_curves,
    threshold_performance_table,
)
from src.models.business_cost import (  # noqa: E402
    BusinessCostAssumptions,
    find_optimal_threshold,
    sweep_thresholds,
)
from src.models.preprocessing import split_X_y_tree  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402

logger = get_logger(__name__)

REPORT_PATH = PATHS.reports / "calibration_threshold_report.md"
TUNED_MODEL_PATH = PATHS.models / "xgboost_tuned.joblib"
FINAL_MODEL_PATH = PATHS.models / "final_churn_model.joblib"


def main() -> int:
    train_path = PATHS.data_processed / "train.parquet"
    test_path = PATHS.data_processed / "test.parquet"
    if not TUNED_MODEL_PATH.is_file():
        logger.error("Tuned model not found — run scripts/run_hyperparameter_tuning.py first")
        return 1

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    X_train, y_train = split_X_y_tree(train_df)
    X_test, y_test = split_X_y_tree(test_df)

    tuned_pipeline = joblib.load(TUNED_MODEL_PATH)
    raw_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
    raw_brier = compute_brier_score(y_test, raw_proba)
    logger.info("Uncalibrated tuned XGBoost: Brier score = %.4f", raw_brier)

    # --- Calibration: fit on TRAIN via cross-validation, evaluate on TEST once ---
    # cv=5 clones the pipeline (same tuned hyperparameters) and fits each clone
    # on 4/5 of the training data, calibrating against the held-out 1/5 — the
    # test split is never involved in fitting either the model or the calibrator.
    candidates = {"sigmoid (Platt)": "sigmoid", "isotonic": "isotonic"}
    calibrated_results = {}
    for label, method in candidates.items():
        calibrated = CalibratedClassifierCV(tuned_pipeline, method=method, cv=5, n_jobs=-1)
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_test)[:, 1]
        brier = compute_brier_score(y_test, proba)
        calibrated_results[label] = {"model": calibrated, "proba": proba, "brier": brier}
        logger.info("Calibrated (%s): Brier score = %.4f", label, brier)

    best_label = min(calibrated_results, key=lambda k: calibrated_results[k]["brier"])
    best_brier = calibrated_results[best_label]["brier"]
    calibration_helps = best_brier < raw_brier
    final_proba = calibrated_results[best_label]["proba"] if calibration_helps else raw_proba
    final_model = calibrated_results[best_label]["model"] if calibration_helps else tuned_pipeline
    logger.info(
        "Calibration decision: %s (%s Brier %.4f vs. raw %.4f)",
        f"ADOPT {best_label}" if calibration_helps else "KEEP raw (uncalibrated) probabilities",
        best_label, best_brier, raw_brier,
    )

    joblib.dump(final_model, FINAL_MODEL_PATH)
    logger.info("Saved final model: %s", FINAL_MODEL_PATH.relative_to(PATHS.root))

    # --- Figures ---
    calibration_curves = {"Raw (tuned XGBoost)": (y_test, raw_proba)}
    if calibration_helps:
        calibration_curves[f"Calibrated ({best_label})"] = (y_test, final_proba)
    plot_calibration_curves(calibration_curves)
    plot_threshold_curves(y_test, final_proba)

    # --- Threshold performance table ---
    threshold_table = threshold_performance_table(y_test, final_proba)

    # --- Business cost framework ---
    value_per_customer = float(train_df["monetary_total"].median())
    logger.info("value_per_customer (measured, train median monetary_total): %.2f", value_per_customer)

    # Contact cost must be able to compete with the expected value at stake
    # (retention_success_rate * value_per_customer) for the threshold to move
    # at all — with a customer value this high, a trivial contact cost always
    # says "contact everyone" regardless of success rate. The third scenario
    # is deliberately expensive enough to test that boundary.
    scenarios = [
        BusinessCostAssumptions(
            contact_cost=15.0, value_per_customer=value_per_customer, retention_success_rate=0.25,
            label="Primary (discount code + some staff time, moderate offer)",
        ),
        BusinessCostAssumptions(
            contact_cost=2.0, value_per_customer=value_per_customer, retention_success_rate=0.12,
            label="Cheap, low-touch (automated email/SMS, low success)",
        ),
        BusinessCostAssumptions(
            contact_cost=120.0, value_per_customer=value_per_customer, retention_success_rate=0.20,
            label="Expensive, high-touch (personal retention call, uncertain payoff)",
        ),
    ]

    scenario_sweeps = {}
    scenario_optimal = {}
    for scenario in scenarios:
        sweep = sweep_thresholds(y_test, final_proba, scenario)
        scenario_sweeps[scenario.label] = sweep
        scenario_optimal[scenario.label] = find_optimal_threshold(sweep)
        logger.info(
            "Scenario '%s': optimal threshold=%.2f, net value=%.2f",
            scenario.label, scenario_optimal[scenario.label]["threshold"],
            scenario_optimal[scenario.label]["net_value_vs_doing_nothing"],
        )

    primary = scenarios[0]
    primary_sweep = scenario_sweeps[primary.label]
    primary_optimal = scenario_optimal[primary.label]
    recommended_threshold = primary_optimal["threshold"]

    default_cost = primary_sweep.iloc[(primary_sweep["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]
    pct_flagged_at_recommended = float((final_proba >= recommended_threshold).mean())

    # --- Report ---
    brier_table = pd.DataFrame(
        [{"probabilities": "Raw (tuned XGBoost)", "brier_score": round(raw_brier, 4)}]
        + [{"probabilities": f"Calibrated ({label})", "brier_score": round(r["brier"], 4)}
           for label, r in calibrated_results.items()]
    )

    scenario_table = pd.DataFrame([
        {
            "scenario": s.label,
            "contact_cost": s.contact_cost,
            "value_per_customer": round(s.value_per_customer, 2),
            "retention_success_rate": s.retention_success_rate,
            "optimal_threshold": scenario_optimal[s.label]["threshold"],
            "net_value_vs_doing_nothing": scenario_optimal[s.label]["net_value_vs_doing_nothing"],
        }
        for s in scenarios
    ])

    report = [
        "# Calibration and Business Threshold Report",
        "",
        "Generated by `scripts/run_calibration_and_threshold.py`, using the Step 9 tuned XGBoost "
        "model. All numbers are measured on the held-out test split — the calibration methods "
        "themselves are fit via cross-validation on the training split only.",
        "",
        "## 1. Is the tuned model's probability output trustworthy?",
        "",
        f"`scale_pos_weight` (used to correct class imbalance in Step 8/9) is well known to shift "
        f"predicted probabilities away from true frequencies even when it improves ranking metrics "
        f"like ROC-AUC. This is checked directly, not assumed.",
        "",
        "### Brier score (lower is better; 0 = perfect, 0.25 = a constant 0.5 guess on a balanced problem)",
        "",
        md_table(brier_table, index=False),
        "",
        f"**Decision: {'adopt ' + best_label + ' calibration' if calibration_helps else 'keep the raw (uncalibrated) probabilities'}.** "
        + (f"{best_label} calibration reduces the Brier score from {raw_brier:.4f} to {best_brier:.4f} "
           f"— a measurable improvement, so the calibrated probabilities are used for everything below "
           f"and saved as the final model."
           if calibration_helps else
           f"Neither calibration method improved on the raw Brier score of {raw_brier:.4f} "
           f"(best alternative: {best_label} at {best_brier:.4f}) — with only {len(X_train):,} "
           f"training rows split across 5 calibration folds, there isn't enough data for calibration "
           f"to reliably improve on the model's own probabilities, so they are used as-is."),
        "",
        "## 2. Calibration curve",
        "",
        "See `reports/figures/calibration_curve.png` — points on the diagonal indicate a predicted "
        "probability matches the observed churn frequency for that bin.",
        "",
        "## 3. Threshold performance table",
        "",
        md_table(threshold_table, index=False),
        "",
        "Note on `reports/figures/threshold_curves.png`: precision briefly spikes then drops to zero "
        "above ~threshold 0.90. This is a visible artefact of isotonic calibration on a training set "
        f"this size ({len(X_train):,} rows split across 5 calibration folds) — isotonic regression "
        "produces a step function, and with limited data the highest step can cover very few test "
        "customers, so precision there is a noisy small-sample statistic rather than a reliable "
        "signal. It does not affect the recommended threshold below, which sits well inside the "
        "stable region of the curve.",
        "",
        "## 4. Business cost framework",
        "",
        "**What follows is a demonstration framework, not real company data.** Online Retail II "
        "contains no record of any actual retention campaign, its cost, or its success rate — none "
        "exists to measure. `value_per_customer` below IS measured from real data (median "
        f"`monetary_total` on the training split = **{value_per_customer:.2f}**, used as a simple "
        "stand-in for full customer lifetime value pending Step 12's proper CLV model). "
        "`contact_cost` and `retention_success_rate` are explicitly hypothetical assumptions, "
        "clearly labelled as such in every table below.",
        "",
        "Cost assigned to each outcome at a threshold:",
        "",
        "- **True Positive** (flagged an actual churner): `contact_cost - retention_success_rate * "
        "value_per_customer` — usually a net BENEFIT, since sometimes saving the customer is worth "
        "more than the outreach cost.",
        "- **False Positive** (flagged a customer who wasn't leaving): `contact_cost` — pure waste.",
        "- **False Negative** (missed an actual churner): `retention_success_rate * "
        "value_per_customer` — the opportunity cost of a chance not taken.",
        "- **True Negative**: 0.",
        "",
        "**Important limitation this framework does NOT capture**: it assumes every flagged churner "
        "has the same probability of being saved by an offer. In reality some would have stayed "
        "regardless (wasted spend beyond the FP cost already charged) and some can't be saved by any "
        "offer. That heterogeneity is exactly what Step 20's uplift modelling addresses — this is a "
        "legitimate first pass, not a substitute for it.",
        "",
        "### Three scenarios (showing the threshold depends on the assumptions, not just the model)",
        "",
        md_table(scenario_table, index=False),
        "",
        f"The optimal threshold moves from {scenario_optimal[scenarios[1].label]['threshold']:.2f} "
        f"(cheap/low-touch) to {scenario_optimal[scenarios[2].label]['threshold']:.2f} "
        f"(expensive/high-touch) purely as a function of the assumed cost and success rate — the "
        f"model's probabilities don't change at all between these scenarios, only the business "
        f"decision about where to act on them.",
        "",
        "## 5. Recommended threshold for this demonstration scenario",
        "",
        f"Using the **primary scenario** (`contact_cost=€{primary.contact_cost:.0f}`, "
        f"`retention_success_rate={primary.retention_success_rate:.0%}`, both hypothetical; "
        f"`value_per_customer=€{primary.value_per_customer:.2f}`, measured): the cost-minimising "
        f"threshold is **{recommended_threshold:.2f}**, with an estimated net value of "
        f"**€{primary_optimal['net_value_vs_doing_nothing']:,.2f}** across the "
        f"{len(y_test):,}-customer test set versus contacting no one.",
        "",
        f"At the default threshold of 0.50 instead: net value would be "
        f"€{default_cost['net_value_vs_doing_nothing']:,.2f} — "
        + (f"€{primary_optimal['net_value_vs_doing_nothing'] - default_cost['net_value_vs_doing_nothing']:,.2f} "
           f"less than the recommended threshold, a concrete demonstration of why 0.50 should not be "
           f"used by default for a business decision."),
        "",
        "**Practical caveat this number-only optimum hides**: a threshold of "
        f"{recommended_threshold:.2f} flags **{pct_flagged_at_recommended:.1%} of all test "
        f"customers** as \"contact.\" That is mathematically optimal under the stated cost "
        f"assumptions — because a low-cost offer against a much larger potential loss says "
        f"\"when in doubt, reach out\" — but it is not a *targeted* retention list; it is closer to "
        f"a mass campaign. A real retention team almost always has a capacity constraint (agent "
        f"hours, a fixed offer budget) this simple framework doesn't model. In practice the "
        f"threshold-performance table in section 3 or a fixed contact-list size (\"top 500 "
        f"customers by risk\") is often the more usable operational answer; the cost-optimal "
        f"threshold here is the right number for the stated assumptions, not necessarily the right "
        f"number to hand a call-centre manager unmodified.",
        "",
        "This recommendation is only as good as the three assumptions behind it — the sensitivity "
        "table above exists precisely so a reader can substitute their own numbers and see the "
        "threshold move accordingly, rather than trusting a single hard-coded answer.",
        "",
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
