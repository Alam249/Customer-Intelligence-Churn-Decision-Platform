"""Tests for src/monitoring.py (Step 19's drift-detection statistics).

PSI and the KS test decide whether a real production system would raise a
drift alarm — a sign error or wrong-axis bug here would either miss a real
shift or cry wolf on a stable population. Each test uses a distribution
where the "right answer" (near-zero for identical samples, clearly large for
an obviously shifted one) is verifiable by construction, not just by running
the code and trusting the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    PSI_MAJOR_THRESHOLD,
    PSI_MODERATE_THRESHOLD,
    categorical_drift_report,
    categorical_psi,
    classify_psi,
    ks_drift_test,
    numeric_drift_report,
    population_stability_index,
)


@pytest.mark.parametrize(
    "psi,expected",
    [
        (0.0, "none"),
        (0.099, "none"),
        (0.10, "moderate"),  # exactly at the moderate threshold -> moderate (inclusive)
        (0.20, "moderate"),
        (0.249, "moderate"),
        (0.25, "major"),  # exactly at the major threshold -> major (inclusive)
        (0.99, "major"),
    ],
)
def test_classify_psi_boundaries(psi, expected):
    assert classify_psi(psi) == expected


def test_thresholds_match_documented_industry_convention():
    """Pins the two constants themselves, not just classify_psi's use of
    them — a future edit to the constants would silently change every
    severity label without this failing unless the constants are checked too.
    """
    assert PSI_MODERATE_THRESHOLD == 0.10
    assert PSI_MAJOR_THRESHOLD == 0.25


def test_psi_near_zero_for_identical_distribution():
    rng = np.random.RandomState(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(0, 1, 2000))  # same distribution, different draw
    psi = population_stability_index(reference, current)
    assert psi < PSI_MODERATE_THRESHOLD


def test_psi_large_for_clearly_shifted_distribution():
    rng = np.random.RandomState(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(3, 1, 2000))  # shifted 3 std devs away
    psi = population_stability_index(reference, current)
    assert psi > PSI_MAJOR_THRESHOLD


def test_psi_ignores_missing_values():
    rng = np.random.RandomState(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current_clean = pd.Series(rng.normal(0, 1, 2000))
    current_with_nans = pd.concat([current_clean, pd.Series([np.nan] * 50)], ignore_index=True)
    assert population_stability_index(reference, current_with_nans) == pytest.approx(
        population_stability_index(reference, current_clean)
    )


def test_psi_returns_zero_for_constant_reference():
    """A constant reference column has no real distribution to bucket into
    deciles — must return 0.0 (nothing to call "shifted"), not raise or NaN.
    """
    reference = pd.Series([5.0] * 100)
    current = pd.Series([5.0, 6.0, 7.0] * 30)
    assert population_stability_index(reference, current) == 0.0


def test_categorical_psi_zero_for_identical_proportions():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 500 + ["B"] * 500)
    assert categorical_psi(reference, current) == pytest.approx(0.0, abs=1e-9)


def test_categorical_psi_large_when_a_category_share_shifts_sharply():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)  # 50/50
    current = pd.Series(["A"] * 950 + ["B"] * 50)  # 95/5
    assert categorical_psi(reference, current) > PSI_MAJOR_THRESHOLD


def test_categorical_psi_handles_a_category_absent_from_current():
    """A category that disappears entirely in `current` must still register
    as drift (via the epsilon floor), not silently drop out of the sum."""
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 1000)  # "B" no longer observed at all
    assert categorical_psi(reference, current) > PSI_MODERATE_THRESHOLD


def test_ks_drift_test_not_drifted_for_same_distribution():
    rng = np.random.RandomState(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(0, 1, 2000))
    result = ks_drift_test(reference, current)
    assert result["drifted"] is False
    assert result["p_value"] > 0.05


def test_ks_drift_test_drifted_for_shifted_distribution():
    rng = np.random.RandomState(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(3, 1, 2000))
    result = ks_drift_test(reference, current)
    assert result["drifted"] is True
    assert result["p_value"] < 0.05
    assert result["ks_statistic"] > 0.5


def test_numeric_drift_report_sorted_by_psi_descending():
    rng = np.random.RandomState(0)
    reference = pd.DataFrame(
        {
            "stable_feature": rng.normal(0, 1, 1000),
            "shifted_feature": rng.normal(0, 1, 1000),
        }
    )
    current = pd.DataFrame(
        {
            "stable_feature": rng.normal(0, 1, 1000),
            "shifted_feature": rng.normal(4, 1, 1000),
        }
    )
    report = numeric_drift_report(reference, current, ["stable_feature", "shifted_feature"])

    assert report.iloc[0]["feature"] == "shifted_feature"
    assert report.iloc[0]["severity"] == "major"
    assert report.iloc[1]["feature"] == "stable_feature"
    assert report.iloc[1]["severity"] == "none"
    assert (report["psi"].diff().dropna() <= 0).all()  # descending order


def test_categorical_drift_report_flags_the_shifted_column():
    reference = pd.DataFrame(
        {
            "stable_country": ["UK"] * 500 + ["FR"] * 500,
            "shifted_country": ["UK"] * 500 + ["FR"] * 500,
        }
    )
    current = pd.DataFrame(
        {
            "stable_country": ["UK"] * 500 + ["FR"] * 500,
            "shifted_country": ["UK"] * 950 + ["FR"] * 50,
        }
    )
    report = categorical_drift_report(reference, current, ["stable_country", "shifted_country"])

    assert report.iloc[0]["feature"] == "shifted_country"
    assert report.iloc[0]["severity"] == "major"
    assert report.loc[report["feature"] == "stable_country", "severity"].item() == "none"
