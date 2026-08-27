"""Uplift modeling / heterogeneous treatment effect estimation (Step 20).

Why this needs simulated data, stated upfront
----------------------------------------------
Steps 6-19 all work with REAL, observed data: real transactions, real churn
labels, real trained-model predictions. Uplift modeling asks a fundamentally
different question — "would CONTACTING this customer change their
behaviour" — which requires knowing what a customer would have done BOTH
with and without treatment. Online Retail II has no retention campaign: no
customer here was ever randomly offered a discount or a retention email, so
there is no real answer to that counterfactual question anywhere in this
dataset.

Every treatment-effect number in this module is therefore SIMULATED, not
measured, and every report/dashboard surface that uses it says so
explicitly. Simulation is the standard way this exact topic is taught when
real experimental data isn't available (it's how `causalml`, `econml`, and
`scikit-uplift`'s own tutorials work) — it lets this project honestly
demonstrate the real technical skill (T-/X-learner implementation, Qini/AUUC
evaluation) without claiming a business finding about real customers that
the data cannot support.

The one thing simulation makes possible that a real experiment never could:
the TRUE per-customer treatment effect is known by construction here, so
model estimates can be validated against ground truth. That is fundamentally
impossible with real experimental data, where only ONE of the two potential
outcomes is ever observed for any one person — the "fundamental problem of
causal inference" (Holland, 1986).

Simulation design
------------------
Treatment is assigned by a fair coin flip, independent of every covariate —
an RCT by construction, so none of the usual confounding concerns apply and
a plain difference in observed outcomes between arms is an unbiased effect
estimate. The TRUE per-customer uplift is a function of the customer's REAL
baseline churn probability (`churn_probability`, the project's actual
trained model output — Step 10), shaped after two documented patterns from
the uplift-modeling literature (Radcliffe & Surry, 2011; Lo, 2002):

  - **Persuadables** (mid-risk customers): the largest positive effect,
    peaking around churn probability ~0.5 — customers who are genuinely on
    the fence respond most to an intervention.
  - **Sleeping dogs** (very low-risk, highly loyal customers): a small
    NEGATIVE effect — contacting an already-happy customer can itself
    prompt them to reconsider the relationship, a real, documented
    phenomenon in retention-campaign literature.
  - **Sure things / lost causes** (very low or very high risk otherwise):
    near-zero effect — already staying, or already effectively gone.

`churn_probability` itself is deliberately NEVER given to the uplift models
as a covariate, even though it's real and available: it was used to build
the simulated ground truth, so including it would let a model partially
"read the answer" rather than learn the relationship from raw customer
behaviour — the harder, more realistic task a real deployment would face,
since no ground-truth uplift score exists to hand a real model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import RANDOM_SEED
from src.eda import CATEGORICAL, INK, save_figure, set_style
from src.models.preprocessing import TREE_CATEGORICAL_FEATURES, TREE_NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

PERSUADABLE_PEAK = 0.50  # baseline churn probability where the positive effect is largest
PERSUADABLE_WIDTH = 0.07  # narrow: true persuadables are a genuine minority (~27% of the
# population exceeds a meaningfully positive effect), not the majority — matching how
# uplift-modeling case studies typically frame the population (Radcliffe & Surry, 2011).
PERSUADABLE_MAGNITUDE = 0.20  # max reduction in churn probability (20pp)

SLEEPING_DOG_PEAK = 0.05  # baseline churn probability of the most loyal customers
SLEEPING_DOG_WIDTH = 0.04
SLEEPING_DOG_MAGNITUDE = 0.05  # max INCREASE in churn probability (5pp) — backfires

TREATMENT_COL = "treatment"
OBSERVED_CHURN_COL = "observed_churn"
TRUE_UPLIFT_COL = "true_uplift"


def true_uplift_function(baseline_churn_proba: np.ndarray) -> np.ndarray:
    """Ground-truth per-customer uplift (reduction in churn probability under
    treatment; positive = treatment helps) as a designed function of the
    REAL baseline churn probability. See module docstring for the rationale.
    """
    p0 = np.asarray(baseline_churn_proba, dtype=float)
    persuadable_bump = PERSUADABLE_MAGNITUDE * np.exp(
        -((p0 - PERSUADABLE_PEAK) ** 2) / (2 * PERSUADABLE_WIDTH**2)
    )
    sleeping_dog_dip = SLEEPING_DOG_MAGNITUDE * np.exp(
        -((p0 - SLEEPING_DOG_PEAK) ** 2) / (2 * SLEEPING_DOG_WIDTH**2)
    )
    return persuadable_bump - sleeping_dog_dip


def simulate_retention_campaign(
    customers: pd.DataFrame,
    baseline_col: str = "churn_probability",
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate a randomised (50/50) retention-campaign RCT on top of the
    REAL customer table. Adds, per customer:

      - `treatment`: 1 if (simulated) treated, 0 if control — Bernoulli(0.5),
        independent of every covariate, so this is a valid RCT by construction.
      - `true_uplift`: the deterministic ground-truth effect (see
        `true_uplift_function`) — known here only because this is a
        simulation; NEVER available in a real deployment.
      - `potential_outcome_control_proba` / `..._treatment_proba`: the churn
        probability this customer would have under each arm.
      - `observed_churn`: the ONE outcome actually "observed" — a Bernoulli
        draw from whichever potential outcome matches the assigned arm. This
        is the only outcome column a real experiment could ever produce.
    """
    if baseline_col not in customers.columns:
        raise KeyError(f"'{baseline_col}' not found — run the full pipeline through Step 10/14 first.")

    rng = np.random.default_rng(random_state)
    out = customers.copy()
    n = len(out)

    p0 = out[baseline_col].to_numpy(dtype=float)
    true_uplift = true_uplift_function(p0)
    p1 = np.clip(p0 - true_uplift, 0.0, 1.0)

    treatment = rng.binomial(1, 0.5, size=n)
    observed_proba = np.where(treatment == 1, p1, p0)
    observed_churn = rng.binomial(1, observed_proba)

    out[TREATMENT_COL] = treatment
    out[TRUE_UPLIFT_COL] = true_uplift
    out["potential_outcome_control_proba"] = p0
    out["potential_outcome_treatment_proba"] = p1
    out[OBSERVED_CHURN_COL] = observed_churn
    return out


def build_uplift_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Median-impute (fit on train only) + one-hot encode — no power
    transform, since both base learners here are tree ensembles, which
    (like Step 8's tree models) don't need one.
    """
    numeric_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    categorical_pipeline = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


# ---------------------------------------------------------------------------
# Uplift learners
# ---------------------------------------------------------------------------
# All three follow sklearn's fit/predict convention (the project's established
# style — see CustomerFeatureEngineer). Uplift here is always defined as
# REDUCTION in churn probability under treatment: positive = good to treat.

_DEFAULT_CLASSIFIER_KWARGS = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 30}
_DEFAULT_REGRESSOR_KWARGS = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 30}


def _default_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=RANDOM_SEED, **_DEFAULT_CLASSIFIER_KWARGS)


def _default_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=RANDOM_SEED, **_DEFAULT_REGRESSOR_KWARGS)


class SLearner:
    """Single model with treatment as an extra feature; uplift is the
    difference between the model's own two counterfactual predictions for
    each customer: `f(X, T=0) - f(X, T=1)`. The simplest possible approach —
    known to be prone to underestimating effects when a flexible model
    otherwise has little incentive to use the treatment feature strongly.
    """

    def __init__(self, base_estimator: object | None = None) -> None:
        self.base_estimator = base_estimator if base_estimator is not None else _default_classifier()

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "SLearner":
        X_aug = np.column_stack([X, treatment])
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_aug, outcome)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        X0 = np.column_stack([X, np.zeros(len(X))])
        X1 = np.column_stack([X, np.ones(len(X))])
        p0 = self.model_.predict_proba(X0)[:, 1]
        p1 = self.model_.predict_proba(X1)[:, 1]
        return p0 - p1


class TLearner:
    """Two independent models, one fit per arm; uplift = mu0(X) - mu1(X).
    Simple and robust to the S-learner's dilution problem, at the cost of
    each model only seeing half the data.
    """

    def __init__(self, base_estimator: object | None = None) -> None:
        self.base_estimator = base_estimator if base_estimator is not None else _default_classifier()

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "TLearner":
        treatment = np.asarray(treatment)
        self.model_treated_ = clone(self.base_estimator).fit(X[treatment == 1], outcome[treatment == 1])
        self.model_control_ = clone(self.base_estimator).fit(X[treatment == 0], outcome[treatment == 0])
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model_treated_.predict_proba(X)[:, 1]
        p0 = self.model_control_.predict_proba(X)[:, 1]
        return p0 - p1


class XLearner:
    """Künzel et al. (2019). Builds on the T-learner's two outcome models,
    then imputes each unit's individual effect using the OTHER arm's model
    (the only way to approximate the unobserved counterfactual for that
    unit), fits an effect-regressor per arm on those imputed effects, and
    combines the two with the propensity score. Under this module's RCT
    (fair coin flip), the propensity is a constant 0.5 for every customer —
    a simplification true real-world uplift modeling can't always assume.
    """

    def __init__(
        self,
        outcome_estimator: object | None = None,
        effect_estimator: object | None = None,
        propensity: float = 0.5,
    ) -> None:
        self.outcome_estimator = outcome_estimator if outcome_estimator is not None else _default_classifier()
        self.effect_estimator = effect_estimator if effect_estimator is not None else _default_regressor()
        self.propensity = propensity

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "XLearner":
        treatment = np.asarray(treatment)
        X_t, y_t = X[treatment == 1], outcome[treatment == 1]
        X_c, y_c = X[treatment == 0], outcome[treatment == 0]

        self.model_treated_ = clone(self.outcome_estimator).fit(X_t, y_t)
        self.model_control_ = clone(self.outcome_estimator).fit(X_c, y_c)

        # Imputed individual effects (churn-reduction convention: + = treatment helped).
        d_treated = self.model_control_.predict_proba(X_t)[:, 1] - y_t
        d_control = y_c - self.model_treated_.predict_proba(X_c)[:, 1]

        self.tau_treated_ = clone(self.effect_estimator).fit(X_t, d_treated)
        self.tau_control_ = clone(self.effect_estimator).fit(X_c, d_control)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        tau1 = self.tau_treated_.predict(X)
        tau0 = self.tau_control_.predict(X)
        return self.propensity * tau0 + (1 - self.propensity) * tau1


def cross_fit_uplift(
    learner_factory,
    preprocessor_factory,
    X_raw: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Out-of-fold uplift predictions for EVERY row via K-fold cross-fitting:
    each row's prediction comes from a model (and a preprocessor) that never
    saw that row during fitting — the same fit-on-train-only discipline as
    everywhere else in this project, just repeated across folds so the full
    population can be evaluated instead of a single held-out slice.

    Why this matters here specifically: individual-level treatment-effect
    estimation needs far more statistical power than average-effect
    estimation. Confirmed directly while building this step — the same true
    uplift score, evaluated on a single 30%-of-4,323 test split, produced a
    non-monotonic decile gradient and a NEGATIVE AUUC purely from sampling
    noise, while evaluating that same score on the full population gave a
    clean gradient and a strongly positive AUUC. Cross-fitting is how a
    modest real population (a few thousand rows, not millions) gets used
    without wasting most of it on a single train/test split.
    """
    treatment, outcome = np.asarray(treatment), np.asarray(outcome)
    out = np.zeros(len(X_raw), dtype=float)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, holdout_idx in kfold.split(X_raw):
        preprocessor = preprocessor_factory()
        X_train = preprocessor.fit_transform(X_raw.iloc[train_idx])
        X_holdout = preprocessor.transform(X_raw.iloc[holdout_idx])

        learner = learner_factory()
        learner.fit(X_train, treatment[train_idx], outcome[train_idx])
        out[holdout_idx] = learner.predict_uplift(X_holdout)
    return out


# ---------------------------------------------------------------------------
# Evaluation: Qini curve, AUUC, uplift-by-decile
# ---------------------------------------------------------------------------


def qini_curve(observed_churn: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> pd.DataFrame:
    """Qini curve (Radcliffe, 2007): cumulative incremental successes from
    targeting the top-ranked-by-`uplift_score` fraction of the population,
    against what random targeting would achieve at the same fraction.

    Only valid because treatment was randomly assigned (an RCT) — under
    random assignment, a simple difference in observed outcomes between
    arms within any subgroup is an unbiased estimate of that subgroup's
    average treatment effect, which is exactly what this curve accumulates.
    "Success" is RETENTION (`1 - observed_churn`), the positive business
    outcome, so the curve reads as "cumulative retained customers gained."
    """
    y = 1 - np.asarray(observed_churn, dtype=float)  # success = retained
    treatment = np.asarray(treatment, dtype=float)
    order = np.argsort(-np.asarray(uplift_score))
    y, treatment = y[order], treatment[order]

    cum_treated = np.cumsum(treatment)
    cum_control = np.cumsum(1 - treatment)
    cum_success_treated = np.cumsum(y * treatment)
    cum_success_control = np.cumsum(y * (1 - treatment))

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(cum_control > 0, cum_treated / cum_control, 0.0)
    qini = cum_success_treated - cum_success_control * scale

    n = np.arange(1, len(y) + 1)
    random_reference = n / len(y) * qini[-1]

    return pd.DataFrame(
        {"n": n, "n_fraction": n / len(y), "qini": qini, "random_reference": random_reference}
    )


def auuc_score(qini: pd.DataFrame) -> float:
    """Area between the Qini curve and the random-targeting reference line,
    normalised by population size. Positive = the ranking beats random
    targeting at identifying who to treat; 0 = no better than random;
    negative = worse than random (actively anti-correlated with true uplift).
    """
    incremental = qini["qini"] - qini["random_reference"]
    return float(np.trapz(incremental, qini["n"]) / len(qini))


def uplift_by_decile(
    observed_churn: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Observed (not predicted) uplift within each predicted-uplift decile —
    valid as a genuine estimate, again because treatment was randomised, so
    the treatment/control outcome gap within any bin is unbiased regardless
    of how that bin was chosen.
    """
    df = pd.DataFrame(
        {
            "retained": 1 - np.asarray(observed_churn, dtype=float),
            "treatment": np.asarray(treatment),
            "score": np.asarray(uplift_score),
        }
    )
    df["decile"] = pd.qcut(df["score"], n_bins, labels=False, duplicates="drop")

    rows = []
    for decile, group in df.groupby("decile"):
        treated, control = group[group["treatment"] == 1], group[group["treatment"] == 0]
        rate_t = treated["retained"].mean() if len(treated) else np.nan
        rate_c = control["retained"].mean() if len(control) else np.nan
        rows.append(
            {
                "decile": int(decile),
                "n_treatment": len(treated),
                "n_control": len(control),
                "retention_rate_treatment": rate_t,
                "retention_rate_control": rate_c,
                "observed_uplift": rate_t - rate_c if len(treated) and len(control) else np.nan,
                "mean_predicted_uplift": group["score"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("decile", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full analysis (shared by scripts/run_uplift_modeling.py and the dashboard's
# uplift-targeting page, so the two can never quietly compute two different
# versions of "the uplift result" — the same principle src/serving.py and
# src/monitoring.py already established for predictions and drift).
# ---------------------------------------------------------------------------

PERSUADABLE_THRESHOLD = 0.08  # true_uplift above this = a genuine persuadable, for precision@k
TOP_K_FRACTION = 0.20
UPLIFT_FEATURE_COLS = TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES


def _preprocessor_factory():
    return build_uplift_preprocessor(TREE_NUMERIC_FEATURES, TREE_CATEGORICAL_FEATURES)


@dataclass
class UpliftAnalysis:
    """Every number Step 20's report and dashboard page need."""

    campaign: pd.DataFrame
    predicted_uplift: dict[str, np.ndarray]
    spread_table: pd.DataFrame
    qini_results: dict[str, pd.DataFrame]
    auuc_table: pd.DataFrame
    ground_truth_correlation: dict[str, float]
    precision_at_k: dict[str, float]
    base_rate: float
    decile_table: pd.DataFrame
    overlap_pct: float
    k: int


def compute_uplift_analysis(
    customers: pd.DataFrame, n_folds: int = 5, random_state: int = RANDOM_SEED
) -> UpliftAnalysis:
    """Simulate the campaign, cross-fit S-/T-/X-learner over the FULL
    population, and compute every evaluation number Step 20 reports.
    `customers` must be the real, fully-scored population (e.g.
    `load_serving_context().customers`).
    """
    campaign = simulate_retention_campaign(customers, random_state=random_state)
    X_raw = campaign[UPLIFT_FEATURE_COLS]
    treatment = campaign["treatment"].to_numpy()
    observed_churn = campaign["observed_churn"].to_numpy()

    learner_factories = {"S-learner": SLearner, "T-learner": TLearner, "X-learner": XLearner}
    predicted_uplift: dict[str, np.ndarray] = {
        name: cross_fit_uplift(
            factory, _preprocessor_factory, X_raw, treatment, observed_churn, n_folds, random_state
        )
        for name, factory in learner_factories.items()
    }

    spread_table = pd.DataFrame(
        {
            "model": [*learner_factories.keys(), "True uplift (ground truth)"],
            "std": [*(predicted_uplift[n].std() for n in learner_factories), campaign["true_uplift"].std()],
            "min": [*(predicted_uplift[n].min() for n in learner_factories), campaign["true_uplift"].min()],
            "max": [*(predicted_uplift[n].max() for n in learner_factories), campaign["true_uplift"].max()],
        }
    ).round(4)

    # Two reference scores, neither a real uplift model — see module docstring.
    predicted_uplift["Risk-based (naive)"] = campaign["churn_probability"].to_numpy()
    predicted_uplift["Oracle (true uplift)"] = campaign["true_uplift"].to_numpy()

    qini_results = {
        name: qini_curve(observed_churn, treatment, score) for name, score in predicted_uplift.items()
    }
    auuc_table = (
        pd.DataFrame(
            {"model": list(qini_results.keys()), "auuc": [auuc_score(q) for q in qini_results.values()]}
        )
        .sort_values("auuc", ascending=False)
        .reset_index(drop=True)
        .round(4)
    )

    true_uplift = campaign["true_uplift"].to_numpy()
    ground_truth_correlation = {
        name: float(np.corrcoef(score, true_uplift)[0, 1])
        for name, score in predicted_uplift.items()
        if name != "Oracle (true uplift)"
    }

    true_persuadable = true_uplift > PERSUADABLE_THRESHOLD
    base_rate = float(true_persuadable.mean())
    k = max(1, int(len(campaign) * TOP_K_FRACTION))
    precision_at_k = {
        name: float(true_persuadable[np.argsort(-score)[:k]].mean())
        for name, score in predicted_uplift.items()
        if name != "Oracle (true uplift)"
    }

    x_learner_rank = pd.Series(predicted_uplift["X-learner"], index=campaign.index).rank(ascending=False)
    priority_rank = campaign["retention_priority_score"].rank(ascending=False)
    top_k_uplift = set(x_learner_rank.nsmallest(k).index)
    top_k_priority = set(priority_rank.nsmallest(k).index)
    overlap_pct = len(top_k_uplift & top_k_priority) / k * 100

    decile_table = uplift_by_decile(observed_churn, treatment, predicted_uplift["X-learner"])

    return UpliftAnalysis(
        campaign=campaign,
        predicted_uplift=predicted_uplift,
        spread_table=spread_table,
        qini_results=qini_results,
        auuc_table=auuc_table,
        ground_truth_correlation=ground_truth_correlation,
        precision_at_k=precision_at_k,
        base_rate=base_rate,
        decile_table=decile_table,
        overlap_pct=overlap_pct,
        k=k,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_qini_curves(qini_curves: dict[str, pd.DataFrame], name: str = "uplift_qini_curves") -> Path:
    """One or more Qini curves overlaid on a single shared random-targeting
    reference line (identical across models since it only depends on the
    population's overall treatment effect, not the ranking).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    reference_drawn = False
    for (name_, qini), color in zip(qini_curves.items(), CATEGORICAL, strict=False):
        ax.plot(qini["n_fraction"], qini["qini"], color=color, linewidth=2, label=name_)
        if not reference_drawn:
            ax.plot(
                qini["n_fraction"],
                qini["random_reference"],
                color=INK["muted"],
                linestyle="--",
                linewidth=1,
                label="Random targeting",
            )
            reference_drawn = True
    ax.set_xlabel("Fraction of population targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative incremental retained customers")
    ax.set_title("Qini curves: model ranking vs. random targeting")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_uplift_by_decile(decile_table: pd.DataFrame, name: str = "uplift_by_decile") -> Path:
    """Observed uplift per predicted-uplift decile — the direct visual
    check of whether the model's ranking tracks REAL treatment/control
    outcome gaps, not just its own predictions.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [f"D{d}" for d in decile_table["decile"]]
    colors = [CATEGORICAL[0] if v >= 0 else CATEGORICAL[1] for v in decile_table["observed_uplift"]]
    ax.bar(x, decile_table["observed_uplift"] * 100, color=colors)
    ax.axhline(0, color=INK["muted"], linewidth=1)
    ax.set_xlabel("Predicted-uplift decile (D9 = highest predicted uplift)")
    ax.set_ylabel("Observed uplift, pp\n(retention rate: treated - control)")
    ax.set_title("Observed uplift by predicted-uplift decile")
    fig.tight_layout()
    return save_figure(fig, name)
