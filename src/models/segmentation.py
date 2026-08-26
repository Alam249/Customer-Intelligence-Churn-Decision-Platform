"""Unsupervised customer segmentation.

Feature choice
--------------
RFM core (`recency_days`, `frequency`, `monetary_total`), `tenure_days`,
catalogue breadth (`distinct_products`, the "service usage" proxy this
dataset supports — see Step 2/6), a trend/engagement signal
(`spend_ratio_90d`), and Step 12's `clv` estimate. All complete for the full
4,323-customer population (verified before this module was written, not
assumed) — no imputation needed.

`recency_score`/`frequency_score`/`monetary_score`/`rfm_score` (Step 6) are
deliberately EXCLUDED here even though they're already RFM-flavoured: they
are discretised (1-5 bucket) versions of features already in this list, and
including both would double-count the same signal in a Euclidean-distance
method exactly the way Step 7 found it distorts a linear model's
coefficients — the concern is different, but the redundancy is the same one.

Preprocessing
-------------
K-Means uses Euclidean distance, so scale matters directly (unlike the
tree-based models in Steps 8-9). The same skew problem Step 7 solved for
Logistic Regression applies here — `monetary_total` and `clv` span orders of
magnitude — so the same fix is reused: Yeo-Johnson power transform +
standardisation, fit on the full population (segmentation has no train/test
split; every customer is scored for the same deployment use case as Step 12).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from src.config import RANDOM_SEED
from src.eda import CATEGORICAL, INK, SEQUENTIAL_BLUE, save_figure, set_style

set_style()

SEGMENTATION_FEATURES = [
    "recency_days", "frequency", "monetary_total", "tenure_days",
    "distinct_products", "spend_ratio_90d", "clv",
]


def build_segmentation_pipeline() -> Pipeline:
    return Pipeline([("power_transform", PowerTransformer(method="yeo-johnson", standardize=True))])


def scan_k_range(X: np.ndarray, k_range: range, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each candidate K.

    Both are reported — neither alone is trusted to pick K. Inertia always
    decreases with K (it's minimised by construction), so it can only reveal
    an "elbow," never an optimum; silhouette can have a spurious global
    maximum at a K too small to be a useful business segmentation.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> KMeans:
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)


def fit_hierarchical(X: np.ndarray, k: int) -> AgglomerativeClustering:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)


def check_stability(X: np.ndarray, k: int, n_splits: int = 5, random_state: int = RANDOM_SEED) -> float:
    """Split-half stability: cluster two independent random halves of the
    population, refit a classifier-free nearest-centroid-style comparison via
    re-clustering both halves independently and score agreement on the
    overlap using Adjusted Rand Index across `n_splits` random splits.

    ARI near 1 means the partition doesn't depend on which half of the data
    was used to find it (stable); near 0 means it does (unstable — likely too
    fine-grained a K, or no real cluster structure to recover).
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    scores = []
    for i in range(n_splits):
        idx = rng.permutation(n)
        half_a, half_b = idx[: n // 2], idx[n // 2:]
        km_a = KMeans(n_clusters=k, random_state=random_state + i, n_init=10).fit(X[half_a])
        km_b = KMeans(n_clusters=k, random_state=random_state + i, n_init=10).fit(X[half_b])
        # Compare on the intersection isn't meaningful for disjoint halves —
        # instead compare each half's own labels against the OTHER half's
        # fitted centroids predicting on it, which is well-defined and tests
        # exactly the stability question: does model A's partition agree with
        # model B's when both see the same (held-out) data?
        pred_a_on_b = km_a.predict(X[half_b])
        scores.append(adjusted_rand_score(km_b.labels_, pred_a_on_b))
    return float(np.mean(scores))


def profile_clusters(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    agg = df.groupby(cluster_col).agg(
        n_customers=("customer_id", "count"),
        recency_days=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary_total=("monetary_total", "median"),
        tenure_days=("tenure_days", "median"),
        distinct_products=("distinct_products", "median"),
        spend_ratio_90d=("spend_ratio_90d", "median"),
        clv=("clv", "median"),
        churn_rate=("is_churned", "mean"),
        churn_probability=("churn_probability", "mean"),
    )
    return agg.round(3)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_elbow_silhouette(scan_table: pd.DataFrame, chosen_k: int, name: str = "kmeans_elbow_silhouette") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scan_table["k"], scan_table["inertia"], marker="o", color=SEQUENTIAL_BLUE)
    axes[0].axvline(chosen_k, color=INK["muted"], linestyle="--", linewidth=1)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")

    axes[1].plot(scan_table["k"], scan_table["silhouette"], marker="o", color=CATEGORICAL[1])
    axes[1].axvline(chosen_k, color=INK["muted"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score")
    fig.tight_layout()
    return save_figure(fig, name)


def plot_cluster_profile_heatmap(df: pd.DataFrame, features: list[str], cluster_col: str = "cluster",
                                  name: str = "cluster_profile_heatmap") -> Path:
    """Standardised (z-score) mean of each feature per cluster — a diverging
    heatmap makes it immediate which clusters are high/low on which
    dimensions, which a table of raw medians does not.
    """
    means = df.groupby(cluster_col)[features].mean()
    z = (means - means.mean()) / means.std()

    fig, ax = plt.subplots(figsize=(0.9 * len(features) + 2, 0.6 * len(means) + 1.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(range(len(means)), [f"Cluster {c}" for c in means.index])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(z.values[i, j]) > 1 else INK["primary"])
    fig.colorbar(im, ax=ax, shrink=0.8, label="Standardised mean (z-score)")
    ax.set_title("Cluster profiles (relative to population mean)")
    fig.tight_layout()
    return save_figure(fig, name)


def plot_cluster_churn_and_value(df: pd.DataFrame, cluster_col: str = "cluster",
                                  name: str = "cluster_churn_value") -> Path:
    """Two separate bar charts (never a dual axis) sharing cluster order."""
    agg = df.groupby(cluster_col).agg(churn_rate=("is_churned", "mean"), clv=("clv", "median")).sort_index()
    labels = [f"Cluster {c}" for c in agg.index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, agg["churn_rate"] * 100, color=CATEGORICAL[7])
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_title("Actual churn rate by cluster")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    axes[1].bar(labels, agg["clv"], color=SEQUENTIAL_BLUE)
    axes[1].set_ylabel("Median CLV (€)")
    axes[1].set_title("Customer value by cluster")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return save_figure(fig, name)
