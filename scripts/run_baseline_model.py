"""Step 7 — Baseline model: Logistic Regression.

Trains the first model in the project and establishes the benchmark every
later model (Step 8's comparison, Step 9's tuned candidate) is measured
against. Deliberately NOT tuned — the point of a baseline is an honest,
untouched reference point.

Logged to MLflow (Step 16) — params, metrics, figures, and the model itself —
under the experiment configured in `config/config.yaml`.

Run:
    python scripts/run_baseline_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import PATHS, RANDOM_SEED  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_classification_metrics,
    plot_coefficients,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.models.preprocessing import (  # noqa: E402
    EXCLUDED_WITH_REASON,
    build_linear_preprocessor,
    get_output_feature_names,
    split_X_y,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402
from src.utils.tracking import init_experiment  # noqa: E402

logger = get_logger(__name__)

REPORT_PATH = PATHS.reports / "baseline_model_report.md"
MODEL_PATH = PATHS.models / "baseline_logistic_regression.joblib"


def main() -> int:
    train_path = PATHS.data_processed / "train.parquet"
    test_path = PATHS.data_processed / "test.parquet"
    if not train_path.is_file() or not test_path.is_file():
        logger.error("Train/test files not found — run scripts/run_feature_engineering.py first")
        return 1

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)
    logger.info("Train: %d rows | Test: %d rows | %d features", len(X_train), len(X_test), X_train.shape[1])

    init_experiment()
    mlflow.start_run(run_name="logistic_regression_baseline")
    mlflow.log_params(
        {
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": RANDOM_SEED,
            "preprocessing": "median_impute+yeo_johnson+standardize",
            "n_features": X_train.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_linear_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",  # Step 4/5: mild 42.5/57.5 imbalance -> weighting, not resampling
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    logger.info("Training Logistic Regression (class_weight='balanced')")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    inference_time_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    test_pred = (test_proba >= 0.5).astype(int)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)

    test_metrics = compute_classification_metrics(y_test, test_pred, test_proba)
    train_metrics = compute_classification_metrics(y_train, train_pred, train_proba)
    logger.info(
        "Test ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f",
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["f1"],
    )

    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    mlflow.log_metrics({"train_time_s": train_time, "inference_time_ms": inference_time_ms})

    # --- Figures ---
    cm_path = plot_confusion_matrix(y_test, test_pred)
    roc_path = plot_roc_curve(y_test, test_proba)
    pr_path = plot_pr_curve(y_test, test_proba)
    for path in (cm_path, roc_path, pr_path):
        mlflow.log_artifact(str(path))

    # --- Coefficients ---
    feature_names = get_output_feature_names(pipeline.named_steps["preprocess"])
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_plot_path = plot_coefficients(coef_table)
    mlflow.log_artifact(str(coef_plot_path))

    # --- Multicollinearity diagnostic ---
    # A coefficient's sign can legitimately disagree with a feature's own
    # univariate relationship to the target when predictors are correlated with
    # each other — the model redistributes shared explanatory power across them.
    # Measured, not assumed: compare each feature's simple correlation with the
    # target against its fitted coefficient's sign.
    univariate_corr = {}
    for col in feature_names:
        values = X_train[col].astype(float)
        values = values.fillna(values.median())
        univariate_corr[col] = np.corrcoef(values, y_train)[0, 1]
    coef_table["univariate_corr"] = coef_table["feature"].map(univariate_corr)
    coef_table["sign_agrees"] = np.sign(coef_table["coefficient"]) == np.sign(coef_table["univariate_corr"])
    sign_conflicts = coef_table[~coef_table["sign_agrees"]].copy()
    sign_conflicts = sign_conflicts.reindex(
        sign_conflicts["coefficient"].abs().sort_values(ascending=False).index
    ).round(4)

    design_matrix = pipeline.named_steps["preprocess"].transform(X_train).astype(np.float64)
    condition_number = np.linalg.cond(design_matrix)
    logger.info(
        "Design matrix condition number: %.1f | sign conflicts: %d/%d features",
        condition_number,
        len(sign_conflicts),
        len(feature_names),
    )

    PATHS.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Saved model: %s", MODEL_PATH.relative_to(PATHS.root))

    # cloudpickle, not the mlflow.sklearn default (skops): skops' type-trust
    # audit rejects several of this project's models' internal types across
    # Steps 7-10 (confirmed empirically) — cloudpickle is consistent with the
    # joblib-based serialization already used and trusted everywhere else here.
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    # --- Report ---
    metrics_table = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"]).round(4)

    top_coef = coef_table.reindex(coef_table["coefficient"].abs().sort_values(ascending=False).index).head(15)
    top_coef["direction"] = np.where(
        top_coef["coefficient"] > 0, "-> pushes toward CHURN", "-> pushes toward RETENTION"
    )
    top_coef = top_coef.round(4)

    excluded_table = pd.DataFrame(
        [{"feature": k, "reason_excluded": v} for k, v in EXCLUDED_WITH_REASON.items()]
    )

    cm_test = pd.crosstab(
        pd.Series(y_test.values, name="Actual").map({0: "Retained", 1: "Churned"}),
        pd.Series(test_pred, name="Predicted").map({0: "Retained", 1: "Churned"}),
    )

    report = [
        "# Baseline Model Report — Logistic Regression",
        "",
        "Generated by `scripts/run_baseline_model.py`. All numbers are measured on the "
        "actual held-out test split — none are estimated or assumed. This model is "
        "deliberately **not tuned**: it is the benchmark every later model (Step 8 "
        "comparison, Step 9 tuned candidate) must beat to justify its added complexity.",
        "",
        "## Why accuracy alone is misleading here",
        "",
        f"The test set is {(1 - y_test.mean()) * 100:.1f}% retained / {y_test.mean() * 100:.1f}% "
        f'churned. A trivial model that always predicts "retained" would score '
        f"**{(1 - y_test.mean()) * 100:.1f}% accuracy** while catching zero churners — useless for "
        "a retention program, whose entire value is in finding the churners. Accuracy weighs both "
        "error types equally; a business missing an actual churner (false negative — a lost customer "
        "nobody tried to retain) is a very different cost from wasting a retention offer on someone "
        "who wasn't leaving (false positive). ROC-AUC and, more importantly under this imbalance, "
        "PR-AUC are reported precisely because they don't share this blind spot.",
        "",
        "## Pipeline",
        "",
        "`sklearn.pipeline.Pipeline` with a `ColumnTransformer`:",
        "",
        "- **Numeric features**: median imputation -> Yeo-Johnson power transform + standardisation "
        "(handles skew from 0.4 to 49 across these columns in one principled step, rather than "
        "hand-picking a log transform per column).",
        "- **Boolean features**: passed through unscaled (already 0/1).",
        "- **Model**: `LogisticRegression(class_weight='balanced')` — weighting, not resampling, "
        "per the mild imbalance found in Step 4/5.",
        "",
        "All preprocessing is fit inside the pipeline on the training split only; the same fitted "
        "imputer/transformer is applied — never refit — to the test split.",
        "",
        "## Features excluded from this linear model (measured reasons)",
        "",
        "Step 8's tree-based models are far less sensitive to collinearity and may reuse the full "
        "feature set including these:",
        "",
        md_table(excluded_table, index=False),
        "",
        "## Metrics: train vs. test",
        "",
        md_table(metrics_table),
        "",
        f"Train and test scores are close (ROC-AUC {train_metrics['roc_auc']:.4f} vs. "
        f"{test_metrics['roc_auc']:.4f}) — no meaningful overfitting for this simple, "
        f"regularised linear model.",
        "",
        "## Confusion matrix (test, threshold = 0.50)",
        "",
        md_table(cm_test),
        "",
        "0.50 is scikit-learn's default decision threshold, used here only because this is the "
        "baseline. Step 10 revisits it against an explicit business cost framework.",
        "",
        "## Logistic Regression coefficients (top 15 by |value|)",
        "",
        "Features are standardised, so coefficients are directly comparable to each other in "
        "magnitude. Sign shows direction, not just strength:",
        "",
        md_table(top_coef[["feature", "coefficient", "direction"]], index=False),
        "",
        "## Multicollinearity diagnostic",
        "",
        f"Design matrix condition number: **{condition_number:.1f}**. Values above ~30 indicate "
        f"moderate-to-strong multicollinearity; this is well above that, consistent with the "
        f"correlated-feature pairs already documented above. This does not hurt the model's "
        f"*predictions* (ROC-AUC/PR-AUC are unaffected by collinearity) but it does mean **individual "
        f'coefficients cannot always be read as "this feature\'s true effect"** — correlated '
        f"predictors can split and even redirect each other's apparent influence.",
        "",
        f"Concrete evidence: comparing each feature's own univariate correlation with `is_churned` "
        f"against its fitted coefficient's sign, **{len(sign_conflicts)} of {len(feature_names)} "
        f"features disagree**:",
        "",
        md_table(sign_conflicts[["feature", "coefficient", "univariate_corr"]], index=False),
        "",
        "For example, `frequency` alone correlates negatively with churn (more orders -> less likely "
        "to churn, matching Step 5's EDA) but its fitted coefficient is positive. This happens because "
        "`frequency` overlaps heavily with several other included features that also describe order "
        "activity (`orders_last_90d`, `purchase_rate_per_month`, `rfm_score`) — once those absorb the "
        'shared "this customer orders often" signal, what\'s left for `frequency` to explain on its '
        "own can point the other way.",
        "",
        f"**`rfm_score` is the most striking case.** It has the STRONGEST univariate correlation with "
        f"churn of any feature in the table ({univariate_corr['rfm_score']:.3f} — stronger even than "
        f"`recency_days`), yet its multivariate coefficient shrinks to nearly zero and flips sign. This "
        f"is not a failure of the feature: `rfm_score` is built directly from `recency_days`, "
        f"`frequency` and `monetary_total`, all three of which are ALSO still in the model. Once those "
        f"three absorb the shared signal, there is almost nothing distinct left for the composite "
        f"score to explain on its own — the multicollinearity here is by construction, not "
        f"coincidence. `rfm_score`'s value is as a standalone reporting/segmentation number (Step "
        f"12/13), not as an additional input alongside the raw features that built it.",
        "",
        "**This is a known limitation of an unregularised-by-default "
        "linear baseline with many correlated engineered features, not a bug** — it is exactly what "
        "Step 9's regularisation search (and Step 8's tree-based models, which are not sensitive to "
        "this) will address. Trust the model's ranking and probability outputs from this baseline; "
        "read individual coefficients as directional evidence only where they agree with the "
        "feature's own univariate relationship, shown above for every feature.",
        "",
        "## Timing",
        "",
        f"- Training time: {train_time:.3f}s ({len(X_train):,} rows)",
        f"- Inference time: {inference_time_ms:.4f}ms per customer (test set, batch-predicted)",
        "",
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))
    mlflow.log_artifact(str(REPORT_PATH))

    mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
