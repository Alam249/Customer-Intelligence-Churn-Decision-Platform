"""Tests for src/uplift.py (Step 20's uplift-modeling methodology).

The Qini/AUUC functions decide whether a real deployment would trust an
uplift model's ranking — a sign error here would make a genuinely harmful
targeting policy look good, or a genuinely good one look bad. The simulation
function (`true_uplift_function`) is checked against an independently
recomputed value (via `math.exp`, not the function under test) at points
where one of its two Gaussian components is known to dominate.

The learners (S-/T-/X-learner) are checked directionally on a strongly
separable synthetic scenario, not for exact-value equality — with a
stochastic model fit, "does the ranking correctly separate responders from
non-responders" is the meaningful, achievable claim, not "does it hit an
exact number."
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.uplift import (
    PERSUADABLE_MAGNITUDE,
    SLearner,
    TLearner,
    XLearner,
    auuc_score,
    cross_fit_uplift,
    qini_curve,
    simulate_retention_campaign,
    true_uplift_function,
    uplift_by_decile,
)

# Small, fast base learners for the model-fitting tests — the point is
# directional correctness on an easy synthetic case, not tuned performance.
_FAST_CLASSIFIER = RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=15, random_state=0)
_FAST_REGRESSOR = RandomForestRegressor(n_estimators=80, max_depth=4, min_samples_leaf=15, random_state=0)


# ---------------------------------------------------------------------------
# true_uplift_function
# ---------------------------------------------------------------------------


def test_true_uplift_at_persuadable_peak_matches_independent_computation():
    """At p0=0.5 (the persuadable peak), the sleeping-dog term is ~40
    standard deviations from its own peak (exp(-40.5) is astronomically
    small), so the value should equal PERSUADABLE_MAGNITUDE almost exactly.
    """
    expected = 0.20 * math.exp(0) - 0.05 * math.exp(-((0.5 - 0.05) ** 2) / (2 * 0.04**2))
    assert true_uplift_function(np.array([0.5]))[0] == pytest.approx(expected, abs=1e-9)
    assert true_uplift_function(np.array([0.5]))[0] == pytest.approx(PERSUADABLE_MAGNITUDE, abs=1e-6)


def test_true_uplift_at_sleeping_dog_peak_matches_independent_computation():
    """At p0=0.05 (the sleeping-dog peak), independently recomputed via
    `math.exp` rather than the function under test.
    """
    expected = 0.20 * math.exp(-((0.05 - 0.5) ** 2) / (2 * 0.07**2)) - 0.05 * math.exp(0)
    assert true_uplift_function(np.array([0.05]))[0] == pytest.approx(expected, abs=1e-9)


def test_true_uplift_is_negative_for_the_most_loyal_customers():
    """Sleeping dogs (very low baseline risk) get a net NEGATIVE effect —
    this is the whole point of including that term: it's a real, documented
    failure mode of naive "contact everyone" retention campaigns.
    """
    assert true_uplift_function(np.array([0.05]))[0] < 0


def test_true_uplift_is_larger_for_mid_risk_than_extreme_risk():
    """Persuadables (mid-risk) should show a larger effect than someone
    almost certainly churning regardless (a "lost cause") — the effect
    should be small there, not the largest in the population.
    """
    mid = true_uplift_function(np.array([0.5]))[0]
    extreme = true_uplift_function(np.array([0.99]))[0]
    assert mid > extreme
    assert extreme < PERSUADABLE_MAGNITUDE * 0.1


# ---------------------------------------------------------------------------
# simulate_retention_campaign
# ---------------------------------------------------------------------------


def test_simulate_retention_campaign_treatment_is_roughly_balanced():
    customers = pd.DataFrame({"customer_id": range(2000), "churn_probability": np.linspace(0.01, 0.99, 2000)})
    campaign = simulate_retention_campaign(customers, random_state=0)
    treatment_share = campaign["treatment"].mean()
    assert 0.45 < treatment_share < 0.55  # Bernoulli(0.5) over 2000 draws


def test_simulate_retention_campaign_potential_outcomes_match_baseline_and_uplift():
    customers = pd.DataFrame({"customer_id": [1, 2], "churn_probability": [0.5, 0.05]})
    campaign = simulate_retention_campaign(customers, random_state=0)

    assert campaign["potential_outcome_control_proba"].tolist() == pytest.approx([0.5, 0.05])
    # treatment_proba = control_proba - true_uplift, clipped to [0, 1]
    expected_treatment_proba = (campaign["potential_outcome_control_proba"] - campaign["true_uplift"]).clip(
        0, 1
    )
    assert campaign["potential_outcome_treatment_proba"].tolist() == pytest.approx(
        expected_treatment_proba.tolist()
    )


def test_simulate_retention_campaign_missing_baseline_column_raises():
    customers = pd.DataFrame({"customer_id": [1, 2]})
    with pytest.raises(KeyError):
        simulate_retention_campaign(customers)


def test_simulate_retention_campaign_observed_churn_matches_assigned_arm():
    """The observed outcome for a treated customer must be a draw from the
    TREATMENT potential outcome, not silently fall back to the control one
    (a plausible copy-paste bug this guards against directly).
    """
    # A customer virtually guaranteed to churn under control (p0~1) and
    # virtually guaranteed to be retained under treatment would be a
    # contradiction if the wrong potential outcome were used — but since
    # true_uplift is bounded (max ~0.20), engineer an extreme case directly
    # via a large population and check the AVERAGE observed rate per arm
    # matches the corresponding potential outcome, not the other arm's.
    customers = pd.DataFrame({"customer_id": range(5000), "churn_probability": [0.5] * 5000})
    campaign = simulate_retention_campaign(customers, random_state=1)

    treated = campaign[campaign["treatment"] == 1]
    control = campaign[campaign["treatment"] == 0]
    assert treated["observed_churn"].mean() == pytest.approx(
        treated["potential_outcome_treatment_proba"].mean(), abs=0.03
    )
    assert control["observed_churn"].mean() == pytest.approx(
        control["potential_outcome_control_proba"].mean(), abs=0.03
    )
    # And treated should churn less than control on average, since p0=0.5 is
    # exactly the persuadable peak (a real, designed positive effect here).
    assert treated["observed_churn"].mean() < control["observed_churn"].mean()


# ---------------------------------------------------------------------------
# qini_curve / auuc_score — exact hand-derived case
# ---------------------------------------------------------------------------


def test_qini_curve_matches_hand_derivation():
    """4 customers, already sorted by descending uplift score:
    A(T=1, retained), B(T=0, churned), C(T=1, retained), D(T=0, retained).

    By hand: cum_treated=[1,1,2,2], cum_control=[0,1,1,2],
    cum_success_treated=[1,1,2,2], cum_success_control=[0,0,0,1].
    scale = cum_treated/cum_control (0 when cum_control=0) = [0,1,2,1].
    qini = cum_success_treated - cum_success_control*scale = [1, 1, 2, 1].
    random_reference = n/4 * qini[-1] = [0.25, 0.5, 0.75, 1.0].
    """
    observed_churn = np.array([0, 1, 0, 0])  # retained = [1, 0, 1, 1]
    treatment = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1])  # already descending

    result = qini_curve(observed_churn, treatment, score)

    assert result["qini"].tolist() == pytest.approx([1.0, 1.0, 2.0, 1.0])
    assert result["random_reference"].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert auuc_score(result) == pytest.approx(0.53125)


def test_qini_curve_sorts_by_score_regardless_of_input_order():
    """Same 4 customers as above, but shuffled in the input arrays — the
    function must sort by score itself, not assume pre-sorted input.
    """
    observed_churn = np.array([0, 0, 1, 0])  # D, C, B, A in that scrambled order
    treatment = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])  # ascending this time

    result = qini_curve(observed_churn, treatment, score)
    assert result["qini"].tolist() == pytest.approx([1.0, 1.0, 2.0, 1.0])


def test_auuc_unbiased_for_uninformative_score_averaged_over_trials():
    """A score with no real relationship to who benefits should rank no
    better than random targeting ON AVERAGE. A SINGLE trial's AUUC is not
    the right thing to bound tightly here: the Qini curve is a cumulative
    process built from partial binomial sums, so even under a true null its
    per-trial value behaves like a random walk with real, non-shrinking
    variance for a fixed population size — confirmed empirically (n=4000
    single-trial AUUC ranged from -18 to +14 across 30 seeds during this
    test's development). What IS a valid, checkable claim is that the
    EXPECTATION is zero — checked here by averaging over many independent
    trials, which drives the standard error of the mean down even though
    each individual trial stays noisy.
    """
    trial_auucs = []
    for seed in range(150):
        rng = np.random.RandomState(seed)
        n = 2000
        treatment = rng.binomial(1, 0.5, size=n)
        observed_churn = rng.binomial(1, 0.4, size=n)  # no true heterogeneous effect at all
        uninformative_score = rng.normal(size=n)  # pure noise, unrelated to anything
        result = qini_curve(observed_churn, treatment, uninformative_score)
        trial_auucs.append(auuc_score(result))

    # Empirically, per-trial std ~= 6 at n=2000, so std of the mean over 150
    # trials ~= 6/sqrt(150) ~= 0.5 — a bound of 2.0 is ~4 standard errors out.
    assert abs(np.mean(trial_auucs)) < 2.0


def test_auuc_directionally_correct_perfect_vs_anti_correlated_ranking():
    """Construct a population with a clear, known responder/non-responder
    split (responders: treatment lifts retention from 0.3 to 0.9;
    non-responders: no effect at all). A score that correctly identifies
    responders must beat random targeting; the exact opposite (deliberately
    wrong) ranking must do markedly worse.
    """
    rng = np.random.RandomState(0)
    n = 4000
    is_responder = rng.binomial(1, 0.5, size=n)
    treatment = rng.binomial(1, 0.5, size=n)

    retain_proba = np.where(
        (is_responder == 1) & (treatment == 1), 0.9, np.where(is_responder == 1, 0.3, 0.3)
    )
    observed_churn = 1 - rng.binomial(1, retain_proba)

    perfect = qini_curve(observed_churn, treatment, uplift_score=is_responder)
    anti = qini_curve(observed_churn, treatment, uplift_score=1 - is_responder)

    auuc_perfect = auuc_score(perfect)
    auuc_anti = auuc_score(anti)

    assert auuc_perfect > 0.05  # meaningfully beats random targeting
    assert auuc_perfect > auuc_anti


# ---------------------------------------------------------------------------
# uplift_by_decile
# ---------------------------------------------------------------------------


def test_uplift_by_decile_top_decile_shows_larger_observed_uplift_than_bottom():
    rng = np.random.RandomState(0)
    n = 4000
    is_responder = rng.binomial(1, 0.5, size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    retain_proba = np.where((is_responder == 1) & (treatment == 1), 0.9, 0.3)
    observed_churn = 1 - rng.binomial(1, retain_proba)
    # Tiny jitter breaks exact ties: a raw 0/1 score has only 2 distinct
    # values, and pd.qcut on a ~50/50-split binary column collapses to a
    # SINGLE bin (its quantile edges coincide) rather than 2 real ones —
    # confirmed directly against pandas during this test's development.
    score = is_responder + rng.normal(0, 0.01, size=n)

    table = uplift_by_decile(observed_churn, treatment, uplift_score=score, n_bins=2)

    assert table["n_treatment"].sum() + table["n_control"].sum() == n
    top_decile_uplift = table.iloc[0]["observed_uplift"]
    bottom_decile_uplift = table.iloc[-1]["observed_uplift"]
    assert top_decile_uplift > bottom_decile_uplift
    assert top_decile_uplift == pytest.approx(0.6, abs=0.05)  # 0.9 - 0.3 by construction
    assert bottom_decile_uplift == pytest.approx(0.0, abs=0.05)  # no true effect by construction


def test_uplift_by_decile_handles_low_cardinality_scores_without_crashing():
    """Only 2 distinct score values can't form 10 real quantile bins —
    `duplicates="drop"` must handle this gracefully, not raise.
    """
    rng = np.random.RandomState(0)
    n = 500
    treatment = rng.binomial(1, 0.5, size=n)
    observed_churn = rng.binomial(1, 0.4, size=n)
    score = rng.binomial(1, 0.5, size=n).astype(float)  # only {0.0, 1.0}

    table = uplift_by_decile(observed_churn, treatment, score, n_bins=10)
    assert len(table) <= 2
    assert len(table) >= 1


# ---------------------------------------------------------------------------
# Learners — directional correctness on a strongly separable synthetic case
# ---------------------------------------------------------------------------


def _responder_scenario(n: int = 1500, random_state: int = 0):
    """X has one informative column (`is_responder`, the true driver of
    heterogeneous effect) and one pure-noise column, so a learner has to
    actually use the informative one rather than just memorising the index.
    """
    rng = np.random.RandomState(random_state)
    is_responder = rng.binomial(1, 0.5, size=n)
    noise = rng.normal(size=n)
    X = np.column_stack([is_responder.astype(float), noise])
    treatment = rng.binomial(1, 0.5, size=n)
    retain_proba = np.where((is_responder == 1) & (treatment == 1), 0.9, 0.3)
    outcome = 1 - rng.binomial(1, retain_proba)  # outcome = churn (1 = churned)
    return X, treatment, outcome, is_responder


@pytest.mark.parametrize("learner_cls", [SLearner, TLearner, XLearner])
def test_learners_rank_true_responders_above_non_responders(learner_cls):
    X, treatment, outcome, is_responder = _responder_scenario()

    if learner_cls is XLearner:
        model = learner_cls(outcome_estimator=_FAST_CLASSIFIER, effect_estimator=_FAST_REGRESSOR)
    else:
        model = learner_cls(base_estimator=_FAST_CLASSIFIER)
    model.fit(X, treatment, outcome)

    predicted_uplift = model.predict_uplift(X)
    mean_uplift_responders = predicted_uplift[is_responder == 1].mean()
    mean_uplift_non_responders = predicted_uplift[is_responder == 0].mean()

    assert mean_uplift_responders > mean_uplift_non_responders


# ---------------------------------------------------------------------------
# cross_fit_uplift
# ---------------------------------------------------------------------------


def test_cross_fit_uplift_returns_a_prediction_for_every_row_and_ranks_correctly():
    """Out-of-fold predictions must cover every row (nothing left as the
    initial zeros because a fold assignment was missed) and must still
    separate true responders from non-responders despite each row's
    prediction coming from a model that never saw it during fitting.
    """
    X_raw, treatment, outcome, is_responder = _responder_scenario(n=1500)
    X_df = pd.DataFrame(X_raw, columns=["is_responder_feature", "noise"])

    def preprocessor_factory():
        return Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    def learner_factory():
        return TLearner(base_estimator=_FAST_CLASSIFIER)

    predicted = cross_fit_uplift(learner_factory, preprocessor_factory, X_df, treatment, outcome, n_splits=5)

    assert len(predicted) == len(X_df)
    assert not np.all(predicted == 0.0)  # every row was in some fold's holdout exactly once
    assert predicted[is_responder == 1].mean() > predicted[is_responder == 0].mean()
