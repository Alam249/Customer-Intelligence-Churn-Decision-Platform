"""Tests for src/models/preprocessing.py (Steps 7-9's feature-set definitions).

The most important property tested here: `get_tree_output_feature_names()`
must return names in EXACTLY the order the fitted `ColumnTransformer` emits
its columns. If that ever drifted, every SHAP value in Step 11's explanations
(and the Step 14 API's `/predict/explain`) would be silently mislabelled —
attached to the wrong feature name while still being numerically "valid,"
which is exactly the kind of bug a human reviewing SHAP plots would never
catch by inspection alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.preprocessing import (
    BOOLEAN_FEATURES,
    EXCLUDED_WITH_REASON,
    NUMERIC_FEATURES,
    TARGET,
    TREE_CATEGORICAL_FEATURES,
    TREE_NUMERIC_FEATURES,
    build_linear_preprocessor,
    build_tree_preprocessor,
    get_output_feature_names,
    get_tree_output_feature_names,
    split_X_y,
    split_X_y_tree,
)

N_ROWS = 30


def _synthetic_df(columns: list[str], target_col: str = TARGET) -> pd.DataFrame:
    """A DataFrame with exactly the given columns, filled with simple
    deterministic synthetic values — built FROM the real column-name
    constants so the test can't silently drift out of sync with them."""
    rng = np.random.RandomState(0)
    data = {}
    for col in columns:
        if col == "country_name":
            data[col] = rng.choice(["United Kingdom", "Germany", "France"], size=N_ROWS)
        elif col in BOOLEAN_FEATURES:
            data[col] = rng.randint(0, 2, size=N_ROWS)
        else:
            data[col] = rng.uniform(1, 100, size=N_ROWS)
    data[target_col] = rng.randint(0, 2, size=N_ROWS)
    data["customer_id"] = range(N_ROWS)
    return pd.DataFrame(data)


def test_linear_and_tree_feature_sets_are_consistent():
    """Every feature excluded from the linear model (for a measured reason,
    Step 7) must (a) genuinely be absent from the linear feature list and
    (b) still be present for tree models — the whole point of the exclusion
    was "linear-specific," not "removed from the project."
    """
    linear_features = set(NUMERIC_FEATURES) | set(BOOLEAN_FEATURES)
    tree_features = set(TREE_NUMERIC_FEATURES) | set(TREE_CATEGORICAL_FEATURES)

    for excluded in EXCLUDED_WITH_REASON:
        assert excluded not in linear_features, (
            f"'{excluded}' is documented as excluded from the linear model but is still in "
            "NUMERIC_FEATURES/BOOLEAN_FEATURES"
        )
        assert excluded in tree_features, (
            f"'{excluded}' is documented as available to tree models but is missing from "
            "TREE_NUMERIC_FEATURES/TREE_CATEGORICAL_FEATURES"
        )

    # Trees are supposed to get a strict superset of the linear feature set
    # (everything linear uses, plus everything excluded from it) — verify
    # that relationship directly rather than assuming it from the exclusion
    # loop above.
    assert linear_features <= tree_features
    assert tree_features - linear_features == set(EXCLUDED_WITH_REASON)


def test_split_X_y_selects_exactly_the_documented_columns():
    df = _synthetic_df(NUMERIC_FEATURES + BOOLEAN_FEATURES)
    df["some_unrelated_column"] = 999  # must be dropped, not accidentally included

    X, y = split_X_y(df)

    assert list(X.columns) == NUMERIC_FEATURES + BOOLEAN_FEATURES
    assert "some_unrelated_column" not in X.columns
    assert y.tolist() == df[TARGET].astype(int).tolist()


def test_split_X_y_tree_selects_exactly_the_documented_columns():
    df = _synthetic_df(TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES)
    df["some_unrelated_column"] = 999

    X, y = split_X_y_tree(df)

    assert list(X.columns) == TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES
    assert "some_unrelated_column" not in X.columns


def test_linear_preprocessor_output_matches_declared_feature_names():
    df = _synthetic_df(NUMERIC_FEATURES + BOOLEAN_FEATURES)
    X, _ = split_X_y(df)

    preprocessor = build_linear_preprocessor()
    transformed = preprocessor.fit_transform(X)
    names = get_output_feature_names(preprocessor)

    assert transformed.shape[1] == len(names)
    assert names == NUMERIC_FEATURES + BOOLEAN_FEATURES


def test_tree_output_feature_names_match_transformed_column_count_and_order():
    """The critical property: `get_tree_output_feature_names` must describe
    the ACTUAL fitted transformer's output, not just a plausible-looking list.
    """
    df = _synthetic_df(TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES)
    X, _ = split_X_y_tree(df)

    preprocessor = build_tree_preprocessor()
    transformed = preprocessor.fit_transform(X)
    names = get_tree_output_feature_names(preprocessor)

    assert transformed.shape[1] == len(names), (
        "get_tree_output_feature_names() length must equal the transformed matrix's column "
        "count, or every downstream SHAP value gets attached to the wrong feature name"
    )
    # Numeric block comes first (declared order), then one-hot country columns.
    assert names[: len(TREE_NUMERIC_FEATURES)] == TREE_NUMERIC_FEATURES

    # The numeric block's VALUES must also line up with the declared order —
    # not just the count. Compare against the imputer's own output directly.
    numeric_only = preprocessor.named_transformers_["numeric"].transform(X[TREE_NUMERIC_FEATURES])
    np.testing.assert_array_almost_equal(transformed[:, : len(TREE_NUMERIC_FEATURES)], numeric_only)

    # One country per observed category, prefixed as documented.
    observed_countries = sorted(X["country_name"].unique())
    expected_country_cols = [f"country_{c}" for c in observed_countries]
    assert names[len(TREE_NUMERIC_FEATURES) :] == expected_country_cols


def test_tree_preprocessor_handles_unseen_category_at_transform_time():
    """`OneHotEncoder(handle_unknown='ignore')` must not raise when a country
    appears at inference time that wasn't in the training data — Step 13's
    segmentation report found exactly this (a country in test not in train)."""
    train_df = _synthetic_df(TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES)
    X_train, _ = split_X_y_tree(train_df)

    preprocessor = build_tree_preprocessor()
    preprocessor.fit(X_train)

    X_test = X_train.copy().iloc[:2]
    X_test.loc[X_test.index[0], "country_name"] = "Nonexistent Country"
    transformed = preprocessor.transform(X_test)  # must not raise
    assert transformed.shape[0] == 2


def test_tree_preprocessor_imputes_missing_values():
    """RandomForestClassifier can't accept NaN — the imputer must actually
    remove them, not just be present in the pipeline unused."""
    df = _synthetic_df(TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES)
    X, _ = split_X_y_tree(df)
    X.loc[X.index[0], TREE_NUMERIC_FEATURES[0]] = np.nan

    preprocessor = build_tree_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed.astype(float)).any()
