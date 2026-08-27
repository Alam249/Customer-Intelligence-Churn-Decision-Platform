"""Hyperparameter search space and Optuna objective for the XGBoost candidate.

Step 8 found XGBoost overfitting substantially with default hyperparameters
(train-test ROC-AUC gap of 0.232 on only 3,458 training rows). Every parameter
searched here is chosen because it directly controls model capacity or adds
regularisation — this is a targeted fix for a diagnosed problem, not a generic
hyperparameter sweep.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.preprocessing import build_tree_preprocessor

# Optuna's TPE sampler exploits the objective's shape after enough trials to
# stop wasting draws on clearly-bad regions — that sample efficiency is the
# reason to prefer it here over RandomizedSearchCV: 8 interacting, mostly
# continuous parameters is exactly the kind of space uniform random search
# covers poorly at a comparable trial budget.
SEARCH_SPACE_DESCRIPTION = {
    "n_estimators": "50-500 (number of trees; more trees = more capacity to overfit)",
    "max_depth": "2-8 (tree depth; Step 8's default of 6 is a likely overfitting driver)",
    "learning_rate": "0.01-0.3, log scale (shrinkage per tree)",
    "min_child_weight": "1-10 (minimum samples-weight per leaf; higher = more conservative splits)",
    "subsample": "0.5-1.0 (row subsampling per tree)",
    "colsample_bytree": "0.5-1.0 (column subsampling per tree)",
    "reg_alpha": "1e-3 to 10.0, log scale (L1 regularisation)",
    "reg_lambda": "1e-3 to 10.0, log scale (L2 regularisation)",
    "gamma": "0-5 (minimum loss reduction required to split further)",
}


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one XGBoost hyperparameter configuration for this trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def build_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_splits: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
):
    """Build the Optuna objective: mean stratified-CV score for one trial's params.

    ``scale_pos_weight`` is fixed at the value derived from the training class
    ratio (as in Step 8) rather than searched — it corrects class imbalance and
    is not a model-capacity knob, so tuning it alongside capacity parameters
    would conflate two different problems.

    The test set never appears here — only `X_train`/`y_train` are cross-
    validated, so no trial can see, let alone optimise against, held-out data.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        pipeline = Pipeline(
            [
                ("preprocess", build_tree_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        **params,
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    return objective
