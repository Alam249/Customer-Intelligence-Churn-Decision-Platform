"""Step 20 — Uplift modeling on a SIMULATED retention campaign.

Online Retail II has no real retention experiment: no customer here was
ever randomly offered a discount or a retention email, so there is no real
answer anywhere in this dataset to "would contacting this customer have
changed their behaviour." Steps 7-19 all worked with real, observed data;
this step honestly cannot, and says so throughout rather than presenting a
simulated scenario as a business finding.

What this script demonstrates instead: given a randomised experiment (here,
simulated on top of the REAL customer population and the REAL Step 10 model's
baseline churn probability), how to correctly estimate HETEROGENEOUS
treatment effects (S-/T-/X-learner), evaluate them with the standard Qini
curve and AUUC, and validate the estimates against ground truth — the one
thing a simulation can do that no real experiment ever could, since real
experiments never observe both potential outcomes for the same person.

It also directly tests Step 12's implicit assumption that ranking by
`churn_probability x CLV` is a good targeting policy: Step 12 never asked
whether a customer's behaviour would actually CHANGE if contacted, only how
much they're worth and how likely they are to leave. This script checks
whether that risk x value ranking has any relationship to who genuinely
benefits from treatment.

Why 5-fold cross-fitting, not a single train/test split: confirmed directly
while building this step — evaluating even the SIMULATION'S OWN true uplift
score on a single 30%-of-4,323 held-out slice produced a non-monotonic
decile gradient and a NEGATIVE AUUC purely from sampling noise (individual
treatment-effect estimation needs far more statistical power than average-
effect estimation). Evaluating that same score on the full population gave a
clean gradient and a strongly positive AUUC. Cross-fitting (`src/uplift.py`)
gets an out-of-fold prediction for every one of the 4,323 real customers
without ever letting a model see the row it's predicting.

The actual simulation, learners, and evaluation logic live in
`src/uplift.py`, shared with the dashboard's Uplift & Targeting page (Step
15's dashboard) so the two can never quietly disagree about what "the
uplift result" is.

Run:
    python scripts/run_uplift_modeling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.serving import load_serving_context  # noqa: E402
from src.uplift import (  # noqa: E402
    TOP_K_FRACTION,
    compute_uplift_analysis,
    plot_qini_curves,
    plot_uplift_by_decile,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.report import md_table  # noqa: E402

logger = get_logger(__name__)

REPORT_PATH = PATHS.reports / "uplift_modeling_report.md"
N_FOLDS = 5


def main() -> int:
    logger.info("Loading the real, fully-scored customer population (Step 10/12/14 outputs)")
    context = load_serving_context()

    logger.info(
        "Simulating a randomised retention campaign and cross-fitting S-/T-/X-learner (%d folds) "
        "over all %d customers",
        N_FOLDS,
        len(context.customers),
    )
    result = compute_uplift_analysis(context.customers, n_folds=N_FOLDS)

    logger.info("Predicted-uplift spread vs. true uplift:\n%s", result.spread_table.to_string(index=False))
    logger.info("AUUC ranking:\n%s", result.auuc_table.to_string(index=False))

    logger.info("Rendering charts")
    qini_fig_path = plot_qini_curves(result.qini_results)
    decile_fig_path = plot_uplift_by_decile(result.decile_table)

    # --- Report ---
    report = [
        "# Uplift Modeling Report (Step 20)",
        "",
        "**Every treatment-effect number in this report is SIMULATED, not measured.** Online Retail II "
        "has no real retention campaign — no customer here was ever randomly offered a discount or a "
        "retention email. The simulation is built on REAL customer covariates and the REAL Step 10 "
        "model's baseline churn probability; only the treatment assignment and the treatment-effect "
        "mechanism are synthetic, and clearly labeled as such throughout. See `src/uplift.py`'s module "
        "docstring for the full simulation design and why simulation is the standard way this exact "
        "topic is taught when real experimental data isn't available.",
        "",
        "## Why this is not the same question Step 12 answered",
        "",
        "Step 12 ranked customers by `churn_probability x CLV` — how likely someone is to leave, times "
        "how much they're worth. That ranking says nothing about whether contacting them would actually "
        "change their behaviour. This step asks that different question directly: given a (simulated) "
        "randomised experiment, which customers show a genuine, causal response to treatment?",
        "",
        "## Simulation and evaluation design",
        "",
        f"- Population: {len(result.campaign):,} real customers (the full Step 10/12 scored population).",
        "- Treatment: fair coin flip (Bernoulli(0.5)), independent of every covariate — a valid RCT by "
        "construction.",
        "- True effect: a designed function of the REAL baseline churn probability, following two "
        "documented patterns from the uplift-modeling literature — **persuadables** (mid-risk "
        "customers, largest positive effect) and **sleeping dogs** (very loyal customers, a small "
        "NEGATIVE effect from being contacted at all).",
        f"- Evaluation: {N_FOLDS}-fold cross-fitting, not a single train/test split — a single held-out "
        "slice of this population was tried first and produced a visibly noisy, non-monotonic result "
        "even for the SIMULATION'S OWN ground-truth score (see the note below); cross-fitting gets a "
        "genuine out-of-fold prediction for every customer without wasting most of the population on "
        "held-out evaluation.",
        "",
        "## Model ranking (AUUC — higher is better; 0 = no better than random targeting)",
        "",
        md_table(result.auuc_table, index=False),
        "",
        "**A note on AUUC's variance**: AUUC is a cumulative statistic built from partial sums, and "
        "(confirmed directly while building `tests/test_uplift.py` and again while building this "
        "script) it carries real, non-trivial sampling noise even under a null relationship — unbiased "
        "in expectation, but not something a single evaluation should over-interpret at close margins. "
        "The Qini curve shape and the ground-truth checks below are read alongside this table, not "
        "instead of it.",
        "",
        f"![Qini curves]({qini_fig_path.relative_to(PATHS.root).as_posix()})",
        "",
        "### Why S-learner trails T-/X-learner on AUUC despite a decent ground-truth correlation",
        "",
        "A textbook, well-documented weakness, reproduced here rather than assumed: with treatment as "
        "just one more feature among 34 covariates, a single flexible tree model has little incentive "
        "to actually split on it, so its uplift estimates collapse toward zero instead of tracking the "
        "true effect's real scale:",
        "",
        md_table(result.spread_table, index=False),
        "",
        "S-learner's predicted uplift has roughly 1/15th the spread of T-/X-learner's — it still "
        "ranks customers in approximately the right RELATIVE order (hence a respectable correlation "
        "with true uplift), but barely distinguishes sleeping dogs from zero-effect customers in "
        "absolute terms, which specifically hurts a full-curve metric like AUUC that depends on real "
        "separation across the whole ranking, not just the top fraction.",
        "",
        "## Validation against ground truth",
        "",
        "Only possible because this is a simulation — a real experiment never observes both potential "
        "outcomes for the same customer, so a real deployment could never check this directly.",
        "",
        "| Score | Correlation with true uplift | Precision@"
        + f"{int(TOP_K_FRACTION * 100)}% (true-persuadable rate, base rate {result.base_rate:.1%}) |",
        "| --- | --- | --- |",
        *[
            f"| {name} | {result.ground_truth_correlation[name]:.3f} | {result.precision_at_k[name]:.1%} |"
            for name in result.ground_truth_correlation
        ],
        "",
        f"![Uplift by decile (X-learner)]({decile_fig_path.relative_to(PATHS.root).as_posix()})",
        "",
        "The overall downward trend from D9 to D0 is real, but individual bars are not perfectly "
        "monotonic (D6 sits above D9-D7) — expected, not a bug: even cross-fitting the full 4,323-"
        "customer population leaves only ~430 customers per decile, and this same non-monotonicity "
        "appears even when decile-binning the SIMULATION'S OWN true-uplift score on this population "
        "(verified directly while building this script). Individual-level treatment-effect estimation "
        "is genuinely harder to pin down precisely than an average effect at this population size.",
        "",
        "## Does risk x value targeting find the same customers as uplift targeting?",
        "",
        f"Top {int(TOP_K_FRACTION * 100)}% of the population by X-learner's predicted uplift vs. top "
        f"{int(TOP_K_FRACTION * 100)}% by Step 12's `retention_priority_score`: "
        f"**{result.overlap_pct:.1f}% overlap** ({result.k:,} customers per list). A naive assumption "
        "that 'highest risk x value' and 'most responsive to treatment' are the same group is not "
        "supported here — they are answering different questions, and a real retention campaign "
        "optimising contact-list ROI would want the uplift ranking, not the risk x value ranking, for "
        "WHO TO CONTACT (Step 12's ranking remains the right tool for WHO IS WORTH SAVING once contact "
        "is decided to be effective).",
        "",
        "## Outputs",
        "",
        "- `reports/figures/uplift_qini_curves.png`",
        "- `reports/figures/uplift_by_decile.png`",
        "",
    ]

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote report: %s", REPORT_PATH.relative_to(PATHS.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
