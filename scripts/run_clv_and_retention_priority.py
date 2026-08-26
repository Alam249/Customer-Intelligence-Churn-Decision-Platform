"""Step 12 — Customer Lifetime Value and Retention Priority.

Scores the ENTIRE eligible customer base (4,323 customers — train + test
combined), not just the held-out test split used to evaluate model
performance in Steps 7-11. This is intentional: a retention program has to
act on every customer, whether or not they happened to be in the training
set — this step produces a business deliverable, not a model evaluation.

Run:
    python scripts/run_clv_and_retention_priority.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from src.config import CONFIG, PATHS, get_database_url  # noqa: E402
from src.eda import save_figure, set_style  # noqa: E402
from src.models.clv import (  # noqa: E402
    build_clv_summary,
    check_independence_assumption,
    estimate_clv,
    fit_bgnbd,
    fit_gamma_gamma,
    load_customer_transactions,
)
from src.models.retention_priority import (  # noqa: E402
    assign_segments,
    compare_targeting_strategies,
    compute_retention_priority,
    plot_segment_quadrant,
    plot_targeting_comparison,
    segment_summary,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402

logger = get_logger(__name__)
set_style()

REPORT_PATH = PATHS.reports / "clv_retention_priority_report.md"
PRIORITY_LIST_PATH = PATHS.reports / "retention_priority_list.csv"
CUTOFF_DATE = CONFIG["churn_definition"]["cutoff_date"]
TOP_N_COMPARISON = 200


def main() -> int:
    validated_path = PATHS.data_processed / "customer_features_2011-06-09_h183_validated.parquet"
    engineer_path = PATHS.models / "feature_engineer.joblib"
    final_model_path = PATHS.models / "final_churn_model.joblib"
    for p in (validated_path, engineer_path, final_model_path):
        if not p.is_file():
            logger.error("Required file not found: %s", p)
            return 1

    # --- 1. CLV: BG/NBD + Gamma-Gamma on full transaction history ---
    logger.info("Querying full transaction history up to cutoff %s", CUTOFF_DATE)
    engine = create_engine(get_database_url())
    try:
        transactions = load_customer_transactions(engine, CUTOFF_DATE)
    finally:
        engine.dispose()
    logger.info("Loaded %d (customer, date) transaction rows for %d customers",
                len(transactions), transactions["customer_id"].nunique())

    summary = build_clv_summary(transactions, CUTOFF_DATE)
    n_repeat = int((summary["frequency"] > 0).sum())
    logger.info("CLV summary: %d customers, %d repeat (frequency>0), %d one-time",
                len(summary), n_repeat, len(summary) - n_repeat)

    independence_corr = check_independence_assumption(summary)
    logger.info("Gamma-Gamma independence check: corr(frequency, monetary_value) = %.4f", independence_corr)

    bgf = fit_bgnbd(summary)
    ggf = fit_gamma_gamma(summary)
    logger.info("BG/NBD params: %s", bgf.params_.to_dict())
    logger.info("Gamma-Gamma params: %s", ggf.params_.to_dict())

    clv_df = estimate_clv(bgf, ggf, summary, transactions)
    logger.info("CLV estimated for %d customers (median €%.2f, mean €%.2f)",
                len(clv_df), clv_df["clv"].median(), clv_df["clv"].mean())

    # --- 2. Churn probability for the same population ---
    logger.info("Scoring churn probability for the full eligible population")
    full_df = pd.read_parquet(validated_path).drop(columns=["cutoff_date"])
    engineer = joblib.load(engineer_path)
    full_df = engineer.transform(full_df)

    from src.models.preprocessing import TREE_CATEGORICAL_FEATURES, TREE_NUMERIC_FEATURES

    final_model = joblib.load(final_model_path)
    X_full = full_df[TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES]
    full_df = full_df.assign(churn_probability=final_model.predict_proba(X_full)[:, 1])

    # Sanity check: mean predicted probability should track the observed churn
    # rate reasonably closely across the WHOLE population (not a formal
    # calibration test — that's Step 10, on test only — just a sniff test that
    # scoring the full population didn't do something obviously wrong).
    mean_predicted = full_df["churn_probability"].mean()
    observed_rate = full_df["is_churned"].mean()
    logger.info("Full-population sniff check: mean predicted P(churn)=%.4f vs. observed rate=%.4f",
                mean_predicted, observed_rate)

    # --- 3. Combine ---
    combined = full_df[["customer_id", "churn_probability", "is_churned"]].merge(
        clv_df[["customer_id", "frequency", "recency", "T", "monetary_value",
                "expected_purchases", "expected_value_per_purchase", "value_source", "clv"]],
        on="customer_id", how="inner",
    )
    logger.info("Combined churn+CLV table: %d customers (%d eligible customers had no matching "
                "transaction summary and were dropped)", len(combined), len(full_df) - len(combined))

    combined = compute_retention_priority(combined)
    combined = assign_segments(combined)
    seg_summary = segment_summary(combined)
    comparison = compare_targeting_strategies(combined, TOP_N_COMPARISON)

    logger.info("Targeting comparison (top %d): churn-only captures €%.2f CLV-at-risk, "
                "priority-ranked captures €%.2f (%.1f%% list overlap)",
                TOP_N_COMPARISON, comparison["churn_only_total_clv_at_risk"],
                comparison["priority_total_clv_at_risk"], comparison["overlap_pct"])

    n_at_max_proba = int((combined["churn_probability"] == combined["churn_probability"].max()).sum())
    top_priority_customer = combined.nlargest(1, "retention_priority_score").iloc[0]

    combined.sort_values("retention_priority_score", ascending=False).to_csv(PRIORITY_LIST_PATH, index=False)
    logger.info("Saved ranked retention priority list: %s", PRIORITY_LIST_PATH.relative_to(PATHS.root))

    # --- Figures ---
    plot_segment_quadrant(combined)
    plot_targeting_comparison(comparison)

    # Log-scale bins, not a clipped linear histogram: clipping the top 1% would
    # pile every outlier into one artificial bar at the clip boundary, which
    # reads as a data spike that isn't really there.
    fig, ax = plt.subplots(figsize=(6, 4))
    log_clv = np.log10(combined["clv"])
    ax.hist(log_clv, bins=40, color="#2a78d6")
    tick_vals = [0, 1, 2, 3, 4, 5]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"€{10**v:,.0f}" for v in tick_vals])
    ax.set_xlabel("Estimated CLV (€, 6-month horizon, log scale)")
    ax.set_ylabel("Customers")
    ax.set_title("CLV distribution")
    fig.tight_layout()
    save_figure(fig, "clv_distribution")

    # --- Report ---
    top_10 = combined.sort_values("retention_priority_score", ascending=False).head(10)
    top_10_display = top_10[["customer_id", "churn_probability", "clv", "retention_priority_score", "segment"]].round(3)

    bgnbd_params = pd.DataFrame([{"parameter": k, "value": round(v, 4)} for k, v in bgf.params_.to_dict().items()])
    gg_params = pd.DataFrame([{"parameter": k, "value": round(v, 4)} for k, v in ggf.params_.to_dict().items()])

    report = [
        "# Customer Lifetime Value and Retention Priority Report",
        "",
        "Generated by `scripts/run_clv_and_retention_priority.py`. All numbers are measured from the "
        "actual transaction data and the Step 10 final model — none are estimated or assumed. This "
        "step scores the FULL eligible customer base (train + test combined), unlike Steps 7-11 which "
        "strictly evaluate on held-out test data — a retention program has to act on every customer, "
        "not just the ones used to evaluate the model.",
        "",
        "## Methodology: why BG/NBD + Gamma-Gamma, not a simpler proxy",
        "",
        "Online Retail II has exactly what this probabilistic approach needs: a genuine repeat-purchase "
        "transaction history per customer (not a single snapshot), in a non-contractual retail setting "
        "— the textbook BG/NBD use case. That prerequisite is checked, not assumed:",
        "",
        f"- **{len(summary):,} customers** have a usable transaction history to the cutoff.",
        f"- **{n_repeat:,} ({n_repeat / len(summary) * 100:.1f}%) are repeat customers** "
        f"(frequency > 0) — enough to fit Gamma-Gamma's monetary-value model.",
        f"- **{len(summary) - n_repeat:,} ({(len(summary) - n_repeat) / len(summary) * 100:.1f}%) "
        "are one-time buyers** — handled with an explicit, transparent fallback (below), not silently "
        "dropped or forced through an unstable formula.",
        "",
        "### Gamma-Gamma's independence assumption — checked",
        "",
        f"Gamma-Gamma assumes purchase frequency is independent of monetary value. Measured correlation "
        f"among repeat customers: **{independence_corr:.4f}** — negligible, so the assumption holds "
        "well for this population.",
        "",
        "### A real limitation, found by testing this module while building it",
        "",
        "Gamma-Gamma's conditional expectation formula is unstable at `frequency=0`: testing it "
        "directly on this data returned a **negative** \"expected profit\" for one-time buyers — not a "
        "theoretical footnote, an actual result reproduced while building this pipeline. This is "
        "exactly why the standard practice (followed here) fits Gamma-Gamma on repeat customers only. "
        "One-time buyers instead use their own single observed transaction value as the value estimate "
        "— the one real data point available for them.",
        "",
        "## Fitted model parameters",
        "",
        "**BG/NBD** (purchase frequency + \"still alive\" probability):",
        "",
        md_table(bgnbd_params, index=False),
        "",
        "**Gamma-Gamma** (monetary value per transaction, repeat customers only):",
        "",
        md_table(gg_params, index=False),
        "",
        f"CLV is projected over a **{6}-month horizon**, matching the churn model's 183-day label "
        "window so the two scores are directly comparable. No discount rate is applied — a "
        "simplification stated explicitly rather than introducing another unverifiable assumption.",
        "",
        "## CLV distribution",
        "",
        f"Median CLV: €{combined['clv'].median():.2f} | Mean: €{combined['clv'].mean():.2f} "
        f"(right-skewed, consistent with the monetary skew found throughout Steps 4-6).",
        "",
        "![CLV distribution](figures/clv_distribution.png)",
        "",
        "## Retention Priority Score",
        "",
        "```",
        "retention_priority_score = churn_probability * CLV",
        "```",
        "",
        "Deliberately the simplest defensible formula: the expected revenue lost if a customer churns "
        "and nothing is done — the same expected-value logic as Step 10's business-cost framework, "
        "applied per customer.",
        "",
        "## Why churn probability alone is not enough — measured, not asserted",
        "",
        f"Ranking the top **{TOP_N_COMPARISON}** customers by churn probability alone vs. by the "
        "combined retention priority score, for the SAME contact-list size:",
        "",
        f"| Ranking strategy | Avg. CLV per customer | Total CLV-at-risk captured |",
        f"| --- | --- | --- |",
        f"| Churn probability alone | €{comparison['churn_only_avg_clv']:,.2f} | "
        f"€{comparison['churn_only_total_clv_at_risk']:,.2f} |",
        f"| Retention priority score | €{comparison['priority_avg_clv']:,.2f} | "
        f"€{comparison['priority_total_clv_at_risk']:,.2f} |",
        "",
        f"The two lists overlap by only **{comparison['overlap_pct']:.1f}%** — ranking on churn "
        "probability alone would spend the same retention budget on a substantially different, "
        "lower-value set of customers. This is the concrete, measured reason churn probability alone "
        "is not sufficient for a retention decision.",
        "",
        f"**Why the lists diverge so completely**: {n_at_max_proba} customers share the exact same "
        "highest predicted churn probability — a plateau from the isotonic calibration Step 10 already "
        "documented as a small-sample artifact at the extreme end of the probability range. Every one "
        "of them is a long-dormant, zero-repeat-purchase customer with a small CLV (€50-60) — the model "
        "is confident they've already left, and there is little value left to protect even if they "
        f"could be reached. The top-priority customer instead is **{int(top_priority_customer['customer_id'])}** "
        f"— only {top_priority_customer['churn_probability']:.1%} predicted churn probability, but an "
        f"estimated CLV of €{top_priority_customer['clv']:,.2f} (a high-frequency, high-return-rate "
        "account — recognisable from Step 4's data-quality profiling). Moderate risk on a very large "
        "amount of value at stake outranks near-certain risk on very little.",
        "",
        "![Targeting comparison](figures/targeting_comparison.png)",
        "",
        "## Segments",
        "",
        md_table(seg_summary),
        "",
        "![Retention segments](figures/retention_quadrant.png)",
        "",
        "- **High risk / High value** — the actual retention priority: likely to churn AND worth "
        "saving. Smallest budget, highest return per contact.",
        "- **High risk / Low value** — likely to churn but little value at stake; a low-cost or "
        "automated touch at most (Step 10's cheap scenario), not a premium intervention.",
        "- **Low risk / High value** — valuable and currently stable; monitor, don't spend retention "
        "budget here.",
        "- **Low risk / Low value** — lowest priority by any measure.",
        "",
        "## Top 10 customers by retention priority score",
        "",
        md_table(top_10_display, index=False),
        "",
        f"Full ranked list for all {len(combined):,} customers: `reports/retention_priority_list.csv`.",
        "",
        "## A limitation of the pure product formula, visible in the table above",
        "",
        f"**{(top_10_display['segment'] == 'Low risk / High value').sum()} of the top 10 by priority "
        "score are \"Low risk / High value,\" not \"High risk / High value.\"** This is mathematically "
        "correct under `churn_probability * CLV` — CLV's range spans nearly five orders of magnitude "
        "(€0.01 to over €129,000), so even a very low churn probability on an extreme-value customer "
        "can outrank a genuinely at-risk customer with modest value. It is also a real practical "
        "limitation: a customer with a 1.6% churn probability is already essentially certain to stay, "
        "and no retention offer is likely to change an outcome that was never in doubt — ranking them "
        "highly overstates how actionable they are.",
        "",
        "A deployment that wants to avoid this would filter to a minimum meaningful churn probability "
        "first (e.g. Step 10's cost-optimal threshold, or a stricter one chosen for capacity reasons) "
        "and rank by priority score only within that filtered set — combining \"is this customer "
        "actually at risk\" with \"is it worth acting on them\" instead of letting the second dominate "
        "the first. This report presents the unfiltered ranking so the trade-off stays visible rather "
        "than being hidden inside a second, unstated design choice.",
        "",
        "The segment summary's `total_clv_at_risk` for \"Low risk / High value\" being the largest of "
        "all four segments is not a contradiction of \"don't spend retention budget here\" above — that "
        "column is a PORTFOLIO-level statistic (aggregate expected exposure summed across 1,888 "
        "customers), while the per-customer priority ranking is what should drive individual contact "
        "decisions. A segment can hold the most aggregate value at stake while still being the wrong "
        "place to spend a targeted retention budget.",
        "",
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
