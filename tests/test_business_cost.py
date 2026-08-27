"""Tests for src/models/business_cost.py (Step 10's threshold framework).

A sign error or swapped term in these formulas would silently flip which
threshold looks "optimal," changing the business recommendation without any
obviously wrong-looking number — exactly the kind of bug worth guarding
against with an exact hand-computed check, not just a smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.business_cost import (
    BusinessCostAssumptions,
    cost_at_threshold,
    find_optimal_threshold,
    sweep_thresholds,
)


def test_cost_properties_match_documented_formulas():
    a = BusinessCostAssumptions(contact_cost=10.0, value_per_customer=100.0, retention_success_rate=0.2)
    assert a.cost_tp == pytest.approx(10.0 - 0.2 * 100.0)  # 10 - 20 = -10 (a net benefit)
    assert a.cost_fp == pytest.approx(10.0)
    assert a.cost_fn == pytest.approx(0.2 * 100.0)  # 20
    assert a.cost_tn == 0.0


def test_cost_at_threshold_confusion_matrix_counts():
    # Hand-constructed: at threshold 0.5, predictions are [1,1,0,0,1,0].
    y_true = np.array([1, 0, 1, 0, 1, 1])
    y_proba = np.array([0.9, 0.6, 0.4, 0.1, 0.8, 0.3])
    assumptions = BusinessCostAssumptions(
        contact_cost=1.0, value_per_customer=100.0, retention_success_rate=0.1
    )

    result = cost_at_threshold(y_true, y_proba, threshold=0.5, assumptions=assumptions)

    # pred = [1,1,0,0,1,0]; true = [1,0,1,0,1,1]
    # TP: idx 0,4 (pred=1,true=1) = 2 | FP: idx 1 (pred=1,true=0) = 1
    # FN: idx 2,5 (pred=0,true=1) = 2 | TN: idx 3 (pred=0,true=0) = 1
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 2
    assert result["tn"] == 1


def test_cost_at_threshold_total_cost_matches_manual_sum():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.1, 0.2])  # pred @ 0.5: [1, 1, 0, 0] -> TP=1, FP=1, FN=1, TN=1
    assumptions = BusinessCostAssumptions(
        contact_cost=5.0, value_per_customer=200.0, retention_success_rate=0.25
    )

    result = cost_at_threshold(y_true, y_proba, threshold=0.5, assumptions=assumptions)

    expected_total = (
        1 * assumptions.cost_tp + 1 * assumptions.cost_fp + 1 * assumptions.cost_fn + 1 * assumptions.cost_tn
    )
    assert result["total_cost"] == pytest.approx(expected_total, abs=0.01)

    # net_value_vs_doing_nothing = (tp+fn)*cost_fn - total_cost
    expected_net_value = (1 + 1) * assumptions.cost_fn - expected_total
    assert result["net_value_vs_doing_nothing"] == pytest.approx(expected_net_value, abs=0.01)


def test_higher_contact_cost_never_decreases_optimal_threshold():
    """A sanity/monotonicity property: making outreach MORE expensive should
    never make the model MORE eager to contact people — the optimal threshold
    should rise or stay flat, never fall, as contact_cost increases (all else
    held equal). This is exactly the qualitative behaviour Step 10's report
    demonstrated with real data; here it's pinned down on a synthetic case.
    """
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=500)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=500) + 0.2, 0, 1)

    cheap = BusinessCostAssumptions(contact_cost=1.0, value_per_customer=500.0, retention_success_rate=0.2)
    expensive = BusinessCostAssumptions(
        contact_cost=300.0, value_per_customer=500.0, retention_success_rate=0.2
    )

    cheap_optimal = find_optimal_threshold(sweep_thresholds(y_true, y_proba, cheap))
    expensive_optimal = find_optimal_threshold(sweep_thresholds(y_true, y_proba, expensive))

    assert expensive_optimal["threshold"] >= cheap_optimal["threshold"]


def test_find_optimal_threshold_picks_minimum_cost_row():
    sweep_table = pd.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3],
            "total_cost": [50.0, -10.0, 30.0],
            "net_value_vs_doing_nothing": [-50.0, 10.0, -30.0],
        }
    )
    optimal = find_optimal_threshold(sweep_table)
    assert optimal["threshold"] == 0.2
    assert optimal["total_cost"] == -10.0


def test_sweep_thresholds_default_range_is_reasonable():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.6, 0.4])
    assumptions = BusinessCostAssumptions(
        contact_cost=1.0, value_per_customer=10.0, retention_success_rate=0.5
    )

    sweep = sweep_thresholds(y_true, y_proba, assumptions)
    assert sweep["threshold"].min() >= 0.0
    assert sweep["threshold"].max() <= 1.0
    assert len(sweep) > 1
