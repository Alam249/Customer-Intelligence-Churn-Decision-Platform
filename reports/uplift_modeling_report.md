# Uplift Modeling Report (Step 20)

**Every treatment-effect number in this report is SIMULATED, not measured.** Online Retail II has no real retention campaign — no customer here was ever randomly offered a discount or a retention email. The simulation is built on REAL customer covariates and the REAL Step 10 model's baseline churn probability; only the treatment assignment and the treatment-effect mechanism are synthetic, and clearly labeled as such throughout. See `src/uplift.py`'s module docstring for the full simulation design and why simulation is the standard way this exact topic is taught when real experimental data isn't available.

## Why this is not the same question Step 12 answered

Step 12 ranked customers by `churn_probability x CLV` — how likely someone is to leave, times how much they're worth. That ranking says nothing about whether contacting them would actually change their behaviour. This step asks that different question directly: given a (simulated) randomised experiment, which customers show a genuine, causal response to treatment?

## Simulation and evaluation design

- Population: 4,323 real customers (the full Step 10/12 scored population).
- Treatment: fair coin flip (Bernoulli(0.5)), independent of every covariate — a valid RCT by construction.
- True effect: a designed function of the REAL baseline churn probability, following two documented patterns from the uplift-modeling literature — **persuadables** (mid-risk customers, largest positive effect) and **sleeping dogs** (very loyal customers, a small NEGATIVE effect from being contacted at all).
- Evaluation: 5-fold cross-fitting, not a single train/test split — a single held-out slice of this population was tried first and produced a visibly noisy, non-monotonic result even for the SIMULATION'S OWN ground-truth score (see the note below); cross-fitting gets a genuine out-of-fold prediction for every customer without wasting most of the population on held-out evaluation.

## Model ranking (AUUC — higher is better; 0 = no better than random targeting)

| model | auuc |
| --- | --- |
| Oracle (true uplift) | 22.9722 |
| X-learner | 12.0118 |
| T-learner | 5.9876 |
| S-learner | 4.1781 |
| Risk-based (naive) | 2.7439 |

**A note on AUUC's variance**: AUUC is a cumulative statistic built from partial sums, and (confirmed directly while building `tests/test_uplift.py` and again while building this script) it carries real, non-trivial sampling noise even under a null relationship — unbiased in expectation, but not something a single evaluation should over-interpret at close margins. The Qini curve shape and the ground-truth checks below are read alongside this table, not instead of it.

![Qini curves](reports/figures/uplift_qini_curves.png)

### Why S-learner trails T-/X-learner on AUUC despite a decent ground-truth correlation

A textbook, well-documented weakness, reproduced here rather than assumed: with treatment as just one more feature among 34 covariates, a single flexible tree model has little incentive to actually split on it, so its uplift estimates collapse toward zero instead of tracking the true effect's real scale:

| model | std | min | max |
| --- | --- | --- | --- |
| S-learner | 0.004 | -0.0054 | 0.0234 |
| T-learner | 0.0596 | -0.1839 | 0.2844 |
| X-learner | 0.0556 | -0.1928 | 0.2532 |
| True uplift (ground truth) | 0.0735 | -0.05 | 0.2 |

S-learner's predicted uplift has roughly 1/15th the spread of T-/X-learner's — it still ranks customers in approximately the right RELATIVE order (hence a respectable correlation with true uplift), but barely distinguishes sleeping dogs from zero-effect customers in absolute terms, which specifically hurts a full-curve metric like AUUC that depends on real separation across the whole ranking, not just the top fraction.

## Validation against ground truth

Only possible because this is a simulation — a real experiment never observes both potential outcomes for the same customer, so a real deployment could never check this directly.

| Score | Correlation with true uplift | Precision@20% (true-persuadable rate, base rate 27.4%) |
| --- | --- | --- |
| S-learner | 0.643 | 69.8% |
| T-learner | 0.512 | 63.9% |
| X-learner | 0.453 | 58.0% |
| Risk-based (naive) | 0.343 | 0.0% |

![Uplift by decile (X-learner)](reports/figures/uplift_by_decile.png)

The overall downward trend from D9 to D0 is real, but individual bars are not perfectly monotonic (D6 sits above D9-D7) — expected, not a bug: even cross-fitting the full 4,323-customer population leaves only ~430 customers per decile, and this same non-monotonicity appears even when decile-binning the SIMULATION'S OWN true-uplift score on this population (verified directly while building this script). Individual-level treatment-effect estimation is genuinely harder to pin down precisely than an average effect at this population size.

## Does risk x value targeting find the same customers as uplift targeting?

Top 20% of the population by X-learner's predicted uplift vs. top 20% by Step 12's `retention_priority_score`: **24.4% overlap** (864 customers per list). A naive assumption that 'highest risk x value' and 'most responsive to treatment' are the same group is not supported here — they are answering different questions, and a real retention campaign optimising contact-list ROI would want the uplift ranking, not the risk x value ranking, for WHO TO CONTACT (Step 12's ranking remains the right tool for WHO IS WORTH SAVING once contact is decided to be effective).

## Outputs

- `reports/figures/uplift_qini_curves.png`
- `reports/figures/uplift_by_decile.png`
