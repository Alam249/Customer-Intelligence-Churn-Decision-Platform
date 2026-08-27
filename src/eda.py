"""Reusable EDA functions: plotting and statistical tests.

Kept outside notebooks/ so the notebook stays a narrative (what we looked at and
what it means) rather than an implementation (how the plot was built). Every
function returns a real result computed from its input — nothing here is
illustrative or fabricated.

Palette: the project's validated colorblind-safe categorical order and the
blue single-hue sequential ramp (see the dataviz skill's reference palette).
Kept to plain hex so this module has no dependency beyond matplotlib/seaborn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import PATHS

# ---------------------------------------------------------------------------
# Palette (light-mode chart surface; static PNGs, so one mode is sufficient)
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = "#2a78d6"
DIVERGING = {"low": "#2a78d6", "mid": "#f0efec", "high": "#e34948"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9"}
SURFACE = "#fcfcfb"

# Binary-outcome charts always use the same two colours in the same order:
# retained (blue, slot 1) vs churned (red, status-critical) — never swapped.
CHURN_COLORS = {False: CATEGORICAL[0], True: STATUS["critical"]}


def set_style() -> None:
    """Apply the project's chart style once per session."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK["muted"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.grid": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "text.color": INK["primary"],
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.facecolor": SURFACE,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure to reports/figures/ with tight bounds; returns the path."""
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    path = PATHS.figures / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def plot_target_balance(y: pd.Series, name: str = "target_balance") -> Path:
    """Horizontal bar of class counts with the rate labelled directly on each bar."""
    counts = y.value_counts().sort_index()
    labels = ["Retained" if not v else "Churned" for v in counts.index]
    colors = [CHURN_COLORS[v] for v in counts.index]
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 2.2))
    bars = ax.barh(labels, counts.values, color=colors, height=0.55)
    for bar, c, p in zip(bars, counts.values, pct.values, strict=True):
        ax.text(
            bar.get_width() + counts.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,} ({p:.1f}%)",
            va="center",
            ha="left",
            color=INK["primary"],
            fontsize=10,
        )
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("Customers")
    ax.set_title("Churn label distribution (183-day horizon, cutoff 2011-06-09)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Numerical features vs churn
# ---------------------------------------------------------------------------


def plot_numeric_by_churn(
    df: pd.DataFrame,
    column: str,
    target: str = "is_churned",
    log_scale: bool = False,
    name: str | None = None,
) -> Path:
    """Box plot of a numeric feature split by churn, with the two fixed churn colours."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    data = [df.loc[~df[target], column].dropna(), df.loc[df[target], column].dropna()]
    bp = ax.boxplot(
        data,
        labels=["Retained", "Churned"],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops={"markersize": 3, "alpha": 0.4},
    )
    for patch, color in zip(bp["boxes"], [CHURN_COLORS[False], CHURN_COLORS[True]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK["primary"])
    for median in bp["medians"]:
        median.set_color(INK["primary"])
        median.set_linewidth(1.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(column + (" (log scale)" if log_scale else ""))
    ax.set_title(f"{column} by churn outcome")
    fig.tight_layout()
    return save_figure(fig, name or f"box_{column}_by_churn")


def plot_binned_churn_rate(
    df: pd.DataFrame, column: str, target: str = "is_churned", bins: int = 5, name: str | None = None
) -> tuple[Path, pd.DataFrame]:
    """Churn rate across equal-sized quantile bins of a numeric feature.

    Reveals monotonic (or non-monotonic) relationships that a single correlation
    coefficient collapses into one number and a box plot doesn't show clearly.
    Returns both the figure path and the underlying table, since the table itself
    is often the citable evidence for a finding.
    """
    binned = pd.qcut(df[column], bins, duplicates="drop")
    table = df.groupby(binned, observed=True)[target].agg(["mean", "size"])
    table.columns = ["churn_rate", "n_customers"]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    labels = [str(i) for i in table.index]
    ax.bar(labels, table["churn_rate"] * 100, color=SEQUENTIAL_BLUE)
    overall = df[target].mean() * 100
    ax.axhline(overall, color=INK["secondary"], linestyle="--", linewidth=1)
    ax.text(
        len(table) - 0.5,
        overall,
        f" overall {overall:.1f}%",
        color=INK["secondary"],
        fontsize=9,
        va="bottom",
        ha="right",
    )
    ax.set_xlabel(f"{column} (quantile bins, low -> high)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(f"Churn rate by {column}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    path = save_figure(fig, name or f"binned_{column}_churn_rate")
    return path, table.round(3)


# ---------------------------------------------------------------------------
# Categorical features vs churn
# ---------------------------------------------------------------------------


def plot_categorical_churn_rate(
    df: pd.DataFrame,
    column: str,
    target: str = "is_churned",
    top_n: int = 8,
    min_count: int = 15,
    name: str | None = None,
) -> Path:
    """Churn rate per category, ordered by rate, restricted to categories with
    enough customers that the rate is not noise. Sample size is shown on each bar
    so a reader cannot mistake a 2-customer category for a robust signal.
    """
    counts = df[column].value_counts()
    eligible = counts[counts >= min_count].index[:top_n]
    rates = df[df[column].isin(eligible)].groupby(column, observed=True)[target].mean().sort_values()
    sizes = counts.loc[rates.index]

    fig, ax = plt.subplots(figsize=(6.5, max(2.4, 0.45 * len(rates))))
    bars = ax.barh(rates.index.astype(str), rates.values * 100, color=SEQUENTIAL_BLUE)
    overall = df[target].mean() * 100
    ax.axvline(overall, color=INK["secondary"], linestyle="--", linewidth=1)
    ax.text(overall, len(rates) - 0.3, f" overall {overall:.1f}%", color=INK["secondary"], fontsize=9)
    for bar, n in zip(bars, sizes.values, strict=True):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center",
            fontsize=8.5,
            color=INK["muted"],
        )
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(f"Churn rate by {column} (categories with >= {min_count} customers)")
    fig.tight_layout()
    return save_figure(fig, name or f"churnrate_{column}")


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], name: str = "correlation_heatmap") -> Path:
    """Diverging blue/red heatmap of pairwise Pearson correlation, gray at zero."""
    corr = df[columns].corr()
    cmap = sns.diverging_palette(220, 15, s=70, l=50, sep=10, as_cmap=True, center="light")
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 2, 0.55 * len(columns) + 1))
    sns.heatmap(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor=SURFACE,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
        annot=len(columns) <= 12,
        fmt=".2f",
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Feature correlation matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def mannwhitney_by_churn(df: pd.DataFrame, column: str, target: str = "is_churned") -> dict[str, Any]:
    """Mann-Whitney U test: does a numeric feature's distribution differ by churn?

    Used instead of a t-test because these features are heavily right-skewed
    (verified in Step 4) — Mann-Whitney compares distributions via ranks and does
    not assume normality.
    """
    a = df.loc[~df[target], column].dropna()
    b = df.loc[df[target], column].dropna()
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "feature": column,
        "median_retained": round(a.median(), 2),
        "median_churned": round(b.median(), 2),
        "u_stat": round(stat, 1),
        "p_value": p,
        "significant_at_0.01": bool(p < 0.01),
    }


def chi2_by_churn(
    df: pd.DataFrame, column: str, target: str = "is_churned", min_count: int = 15
) -> dict[str, Any]:
    """Chi-square test of independence between a category and churn.

    Rare categories (fewer than ``min_count`` customers) are pooled into 'Other'
    first, since sparse cells make the chi-square approximation unreliable.
    """
    counts = df[column].value_counts()
    grouped = df[column].where(df[column].isin(counts[counts >= min_count].index), "Other")
    table = pd.crosstab(grouped, df[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "feature": column,
        "categories_tested": int(table.shape[0]),
        "chi2": round(chi2, 2),
        "dof": int(dof),
        "p_value": p,
        "significant_at_0.01": bool(p < 0.01),
    }
