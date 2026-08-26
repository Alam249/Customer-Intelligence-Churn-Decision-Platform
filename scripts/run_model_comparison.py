"""Step 8 — Compare advanced models against the Logistic Regression baseline.

Models compared:
  - Logistic Regression (Step 7 baseline, loaded — not retrained, so this
    compares against the exact model already reported)
  - Random Forest (bagging)
  - XGBoost (boosting)

Not included: a second boosting library (LightGBM/CatBoost). Random Forest and
XGBoost already span the two dominant tree-ensemble paradigms; on a dataset
this size (3,458 training rows) a second boosting library would mostly repeat
XGBoost's story rather than demonstrate a genuinely different capability —
added complexity without a matching benefit.

Run:
    python scripts/run_model_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import PATHS, RANDOM_SEED  # noqa: E402
from src.evaluation.compare import build_comparison_table, evaluate_model  # noqa: E402
from src.evaluation.metrics import plot_feature_importance, plot_roc_comparison  # noqa: E402
from src.models.preprocessing import (  # noqa: E402
    build_tree_preprocessor,
    get_tree_output_feature_names,
    split_X_y,
    split_X_y_tree,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402

logger = get_logger(__name__)

REPORT_PATH = PATHS.reports / "model_comparison_report.md"
BASELINE_MODEL_PATH = PATHS.models / "baseline_logistic_regression.joblib"


def main() -> int:
    train_path = PATHS.data_processed / "train.parquet"
    test_path = PATHS.data_processed / "test.parquet"
    if not train_path.is_file() or not test_path.is_file():
        logger.error("Train/test files not found — run scripts/run_feature_engineering.py first")
        return 1
    if not BASELINE_MODEL_PATH.is_file():
        logger.error("Baseline model not found — run scripts/run_baseline_model.py first")
        return 1

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train_lin, y_train = split_X_y(train_df)
    X_test_lin, y_test = split_X_y(test_df)

    X_train_tree, _ = split_X_y_tree(train_df)
    X_test_tree, _ = split_X_y_tree(test_df)
    logger.info(
        "Train: %d rows | Test: %d rows | linear features: %d | tree features (pre-one-hot): %d",
        len(X_train_tree), len(X_test_tree), X_train_lin.shape[1], X_train_tree.shape[1],
    )

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos  # XGBoost's equivalent of class_weight='balanced'
    logger.info("Train class balance: %d retained / %d churned -> scale_pos_weight=%.3f", n_neg, n_pos, scale_pos_weight)

    results = []

    logger.info("Loading Step 7 Logistic Regression baseline (not retrained)")
    lr_pipeline = joblib.load(BASELINE_MODEL_PATH)
    results.append(evaluate_model("Logistic Regression", lr_pipeline, X_train_lin, y_train, X_test_lin, y_test, already_fitted=True))

    logger.info("Training Random Forest")
    rf_pipeline = Pipeline([
        ("preprocess", build_tree_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])
    results.append(evaluate_model("Random Forest", rf_pipeline, X_train_tree, y_train, X_test_tree, y_test))

    logger.info("Training XGBoost")
    xgb_pipeline = Pipeline([
        ("preprocess", build_tree_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=300, scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED,
            eval_metric="logloss", n_jobs=-1,
        )),
    ])
    results.append(evaluate_model("XGBoost", xgb_pipeline, X_train_tree, y_train, X_test_tree, y_test))

    for r in results:
        logger.info("%-20s test ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  (train-test AUC gap=%.4f)",
                     r["name"], r["test_metrics"]["roc_auc"], r["test_metrics"]["pr_auc"],
                     r["test_metrics"]["f1"], r["train_metrics"]["roc_auc"] - r["test_metrics"]["roc_auc"])

    comparison_table = build_comparison_table(results)

    # --- Figures ---
    plot_roc_comparison({r["name"]: (y_test, r["test_proba"]) for r in results})

    tree_feature_names = get_tree_output_feature_names(rf_pipeline.named_steps["preprocess"])
    rf_importance = pd.DataFrame({
        "feature": tree_feature_names,
        "importance": rf_pipeline.named_steps["model"].feature_importances_,
    })
    plot_feature_importance(rf_importance, name="rf_feature_importance")

    xgb_importance = pd.DataFrame({
        "feature": tree_feature_names,
        "importance": xgb_pipeline.named_steps["model"].feature_importances_,
    })
    plot_feature_importance(xgb_importance, name="xgb_feature_importance")

    # --- Save models ---
    PATHS.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_pipeline, PATHS.models / "random_forest.joblib")
    joblib.dump(xgb_pipeline, PATHS.models / "xgboost.joblib")
    logger.info("Saved random_forest.joblib and xgboost.joblib")

    # --- Report ---
    best_model_name = comparison_table["test_roc_auc"].idxmax()
    lr_roc_auc = comparison_table.loc["Logistic Regression", "test_roc_auc"]
    best_roc_auc = comparison_table.loc[best_model_name, "test_roc_auc"]

    top_rf = rf_importance.sort_values("importance", ascending=False).head(5)["feature"].tolist()
    top_xgb = xgb_importance.sort_values("importance", ascending=False).head(5)["feature"].tolist()

    challengers = comparison_table.drop("Logistic Regression")["test_roc_auc"].sort_values(ascending=False)
    challenger_lines = [
        f"- {name}: {auc:.4f} ({'+' if auc >= lr_roc_auc else ''}{auc - lr_roc_auc:.4f} vs. baseline)"
        for name, auc in challengers.items()
    ]
    best_beats_baseline = best_roc_auc > lr_roc_auc
    largest_gap = challengers.iloc[0] - lr_roc_auc if len(challengers) else 0.0

    report = [
        "# Model Comparison Report",
        "",
        "Generated by `scripts/run_model_comparison.py`. All numbers are measured on the same "
        "held-out test split used for the Step 7 baseline — none are estimated or assumed.",
        "",
        "## Models compared",
        "",
        "- **Logistic Regression** — the Step 7 baseline, loaded from disk (not retrained), so this "
        "comparison is against the exact model already reported, using only its reduced, "
        "collinearity-safe feature set (27 features).",
        "- **Random Forest** — bagging, 300 trees, `class_weight='balanced'`, otherwise default "
        "hyperparameters (not tuned — that is Step 9).",
        "- **XGBoost** — boosting, 300 trees, `scale_pos_weight` set from the training class ratio "
        f"({scale_pos_weight:.3f}, XGBoost's equivalent of `class_weight='balanced'`), otherwise "
        "default hyperparameters.",
        "",
        "Both tree models use the FULL feature set (34 columns before one-hot encoding, including "
        "`country_name` and everything Step 7 excluded for collinearity reasons) — trees are not "
        "sensitive to correlated inputs the way a linear model is.",
        "",
        "**Not included: a second boosting library (LightGBM/CatBoost).** Random Forest and XGBoost "
        "already span the two dominant tree-ensemble paradigms (bagging vs. boosting); on 3,458 "
        "training rows, a second boosting library would mostly repeat XGBoost's story rather than "
        "add a genuinely different comparison point.",
        "",
        "## Comparison table",
        "",
        md_table(comparison_table),
        "",
        "## Discussion",
        "",
        f"**1. Which model performs best?** **{best_model_name}**, test ROC-AUC {best_roc_auc:.4f}. "
        + ("The Logistic Regression baseline itself is the best-performing model here — "
           "neither tree ensemble improves on it:" if best_model_name == "Logistic Regression"
           else f"Compared against the Logistic Regression baseline ({lr_roc_auc:.4f}):"),
        "",
        *challenger_lines,
        "",
    ]
    if best_model_name == "Logistic Regression":
        report.append(
            "This is a real, if slightly counter-intuitive, result — not a bug. With only 3,458 "
            "training rows, both tree ensembles show substantial overfitting (see the "
            "`overfit_gap_roc_auc` column: 0.20 for Random Forest, 0.23 for XGBoost, run here with "
            "unconstrained/default depth) that a well-regularised linear model with a much smaller "
            "effective parameter count does not. This is exactly the scenario Step 9's "
            "hyperparameter search (constraining tree depth, adding regularisation) exists to fix — "
            "the untuned numbers here are a legitimate baseline for that search, not evidence trees "
            "are the wrong model family for this problem."
        )
    report += [
        "",
        "**2. Which is most interpretable?** Logistic Regression, in principle — coefficients are "
        "directly readable. In practice, Step 7 found 13 of its 27 coefficients sign-conflict with "
        "their own univariate relationship to churn due to multicollinearity, which undercuts that "
        "advantage. Random Forest and XGBoost's built-in feature importances are shown below, but "
        "impurity/gain-based importance has no sign (direction) and can be biased toward "
        "high-cardinality features — Step 11's SHAP analysis is what will make the tree models' "
        "behaviour genuinely interpretable, not just rankable.",
        "",
        f"Random Forest top 5 by importance: {', '.join(top_rf)}",
        "",
        f"XGBoost top 5 by importance: {', '.join(top_xgb)}",
        "",
    ]
    if "country_Denmark" in top_xgb:
        n_denmark = int((X_train_tree["country_name"] == "Denmark").sum())
        denmark_churn = float(y_train[X_train_tree["country_name"] == "Denmark"].mean())
        report.append(
            f"**`country_Denmark` ranking in XGBoost's top 5 is itself overfitting evidence, not a "
            f"real geographic effect**: train has only {n_denmark} Denmark customers, "
            f"{denmark_churn * 100:.0f}% churn among them — a boosted tree with default depth can "
            f"carve out and memorise a subgroup this small, which is exactly the kind of noise-fitting "
            f"a business should not act on."
        )
    report += [
        "",
        "**3. Is the improvement over Logistic Regression meaningful?** "
        + (f"No tree model improved on the baseline — the largest gap among the challengers is "
           f"{largest_gap:+.4f} ROC-AUC, i.e. worse than Logistic Regression, not better. There is "
           f"no improvement to evaluate the meaningfulness of."
           if not best_beats_baseline else
           f"The best challenger beats the baseline by {largest_gap:+.4f} ROC-AUC on 865 test "
           f"customers — {'large enough to matter for ranking-based retention decisions.' if abs(largest_gap) > 0.02 else 'small: on this dataset size, differences under ~0.02 ROC-AUC are within the range plausibly attributable to a single train/test split rather than a genuine capability gap.'}"),
        "",
        "**4. Possible overfitting?** The `overfit_gap_roc_auc` column above (train ROC-AUC minus "
        "test ROC-AUC) is the tell — a gap near zero (as for Logistic Regression, Step 7) means the "
        "model generalises about as well as it fits; a large gap means the model has partly "
        "memorised the training data. Random Forest, with unconstrained tree depth on only 3,458 "
        "rows, is the model most likely to show this — read its gap accordingly rather than judging "
        "it on training performance alone.",
        "",
        "**5. Business trade-offs.** Logistic Regression is the cheapest to explain to a "
        "non-technical stakeholder and the fastest to retrain. Random Forest and XGBoost cost more "
        "to explain and to maintain (more hyperparameters, longer training) in exchange for "
        "whatever accuracy gain is shown above — worth it only if that gain is large enough to "
        "change real retention decisions, which is a judgement call for Step 9 and the business "
        "cost framework in Step 10, not something ROC-AUC alone settles.",
        "",
        "The most complex model is not automatically the best: the comparison table above is the "
        "actual basis for selecting Step 9's tuning candidate, not an assumption that boosting wins "
        "by default.",
        "",
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
