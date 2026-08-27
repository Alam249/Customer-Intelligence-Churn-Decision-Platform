"""Inline (non-file-saving) chart renderers for interactive dashboard widgets.

Everything elsewhere in the dashboard reuses the actual PNGs already produced
by Steps 5-13 (`st.image` on `reports/figures/*.png`) — those are real,
already-computed project outputs, and re-rendering them would just be
duplicated work. This module exists ONLY for the one thing that has to be
computed fresh on every user interaction: the threshold explorer on the Model
Performance page, where saving a new PNG to disk on every slider move would
be wasteful and slow. Same palette and conventions as `src/eda.py`, so a
chart drawn here reads as the same visual system as everything static.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.eda import CATEGORICAL, CHURN_COLORS, INK, SEQUENTIAL_BLUE, STATUS, set_style
from src.monitoring import PSI_MAJOR_THRESHOLD, PSI_MODERATE_THRESHOLD

set_style()


def render_confusion_matrix(y_true, y_pred) -> plt.Figure:
    """2x2 confusion matrix at whatever threshold the caller already applied."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap=plt.cm.Blues, vmin=0)
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else INK["primary"]
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    fig.tight_layout()
    return fig


def render_local_shap_bar(
    top_risk_factors: list[dict], top_protective_factors: list[dict], customer_id: int, probability: float
) -> plt.Figure:
    """Local SHAP explanation, rendered in-memory for `st.pyplot` — deliberately
    NOT written to `reports/figures/`. `explain_customer()`'s own
    `save_plot=True` path writes a PNG per call, which would litter that
    directory with one file per customer a dashboard user happens to browse
    to; this mirrors that function's exact styling without the disk write.
    """
    factors = top_risk_factors + top_protective_factors
    factors = sorted(factors, key=lambda f: abs(f["shap_value"]))
    labels = [f"{f['feature']} = {f['value']:.3g}" for f in factors]
    values = [f["shap_value"] for f in factors]
    colors = [CHURN_COLORS[True] if v > 0 else CHURN_COLORS[False] for v in values]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(factors))))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color=INK["muted"], linewidth=1)
    ax.set_xlabel("SHAP value (log-odds)")
    ax.set_title(f"Customer {customer_id} — churn probability {probability:.1%}")
    fig.tight_layout()
    return fig


def render_risk_band_bar(counts: dict[str, int]) -> plt.Figure:
    """Horizontal bar of customer counts per risk band, fixed Low/Medium/High
    order and the project's status-style colouring (not the churn red/blue
    binary, since this is a three-level band, not a churned/retained split).
    """
    order = ["Low", "Medium", "High"]
    colors = {"Low": "#0ca30c", "Medium": "#fab219", "High": CHURN_COLORS[True]}
    values = [counts.get(k, 0) for k in order]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    bars = ax.barh(order, values, color=[colors[k] for k in order])
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,}",
            va="center",
            fontsize=10,
            color=INK["primary"],
        )
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("Customers")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def render_feature_psi_bar(drift_report, top_n: int = 15) -> plt.Figure:
    """Step 19's top-N-by-PSI bar, in-memory — same styling as
    `src.monitoring.plot_feature_psi_bar`, which instead saves the PNG
    `scripts/run_drift_monitoring.py`'s report links to. The dashboard
    recomputes the drift analysis on every page load (cached), so re-drawing
    here rather than depending on that PNG being fresh is the same principle
    `render_confusion_matrix` already applies to the threshold explorer.
    """
    top = drift_report.sort_values("psi", ascending=False).head(top_n).iloc[::-1]
    severity_colors = {"none": CATEGORICAL[0], "moderate": STATUS["warning"], "major": STATUS["critical"]}
    bar_colors = [severity_colors[s] for s in top["severity"]]

    fig, ax = plt.subplots(figsize=(6.5, max(2.4, 0.35 * len(top))))
    ax.barh(top["feature"], top["psi"], color=bar_colors)
    ax.axvline(PSI_MODERATE_THRESHOLD, color=INK["muted"], linestyle="--", linewidth=1)
    ax.axvline(PSI_MAJOR_THRESHOLD, color=INK["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title(f"Feature drift: top {len(top)} by PSI (dashed = moderate/major thresholds)")
    fig.tight_layout()
    return fig


def render_probability_drift(reference_proba, current_proba) -> plt.Figure:
    """Step 19's reference-vs-current churn-probability overlay, in-memory —
    see `render_feature_psi_bar` for why the dashboard redraws rather than
    reusing the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(
        reference_proba,
        bins=25,
        range=(0, 1),
        density=True,
        alpha=0.55,
        color=SEQUENTIAL_BLUE,
        label=f"Reference (n={len(reference_proba):,})",
    )
    ax.hist(
        current_proba,
        bins=25,
        range=(0, 1),
        density=True,
        alpha=0.55,
        color=STATUS["critical"],
        label=f"Current (n={len(current_proba):,})",
    )
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction drift: churn probability distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_qini_curves(qini_curves) -> plt.Figure:
    """Step 20's Qini-curve overlay, in-memory — see `render_feature_psi_bar`
    for why the dashboard redraws rather than reusing the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    reference_drawn = False
    for (name, qini), color in zip(qini_curves.items(), CATEGORICAL, strict=False):
        ax.plot(qini["n_fraction"], qini["qini"], color=color, linewidth=2, label=name)
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
    return fig


def render_uplift_by_decile(decile_table) -> plt.Figure:
    """Step 20's observed-uplift-by-decile bar chart, in-memory — see
    `render_feature_psi_bar` for why the dashboard redraws rather than
    reusing the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [f"D{d}" for d in decile_table["decile"]]
    colors = [CATEGORICAL[0] if v >= 0 else CATEGORICAL[1] for v in decile_table["observed_uplift"]]
    ax.bar(x, decile_table["observed_uplift"] * 100, color=colors)
    ax.axhline(0, color=INK["muted"], linewidth=1)
    ax.set_xlabel("Predicted-uplift decile (D9 = highest predicted uplift)")
    ax.set_ylabel("Observed uplift, pp\n(retention rate: treated - control)")
    ax.set_title("Observed uplift by predicted-uplift decile")
    fig.tight_layout()
    return fig
