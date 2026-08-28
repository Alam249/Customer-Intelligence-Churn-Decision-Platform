# Final Data Science Audit (Step 23)

A skeptical, final re-check of the project — not a rubber stamp. This report covers one substantial NEW empirical analysis (a real holdout validation of the CLV model's forecasts, never done in Steps 1-22) plus a checklist re-verification of reproducibility, leakage prevention, and statistical rigor across the whole project.

## 1. Reproducibility — re-checked, confirmed clean

Every stochastic operation in `src/` and `scripts/` (`train_test_split`, `KFold`, `RandomForestClassifier/Regressor`, `XGBClassifier`, `KMeans`, the Step 20 simulation) was grepped for `random_state`/`RANDOM_SEED` usage: all 32 call sites pass it explicitly, all traceable to the single `RANDOM_SEED=42` in `config/config.yaml`. No raw, unseeded `np.random.*` call exists anywhere in the codebase. `AgglomerativeClustering` (Step 13's cross-algorithm check) needs no seed — Ward linkage is deterministic.

## 2. Data leakage — re-verified, no new issues found

Re-confirmed rather than re-argued: the 4 SQL assertions (Step 3), the fit-on-train-only discipline verified with a real bug-simulation (Step 6, `test_is_high_value_uses_train_threshold_not_test`), and the CalibratedClassifierCV's `cv=5` running on the training split only (Step 10) together cover every place features, calibration, or thresholds could see test-set information.

**One real caveat surfaced, not previously named**: the SAME fixed test split (Step 6) was evaluated and reported at every step from 7 through 15 — the baseline, three-model comparison, tuning, calibration, SHAP, CLV, segmentation, and the API/dashboard. No re-tuning ever used test performance (each step's decision to proceed was based on train-vs-test overfit gaps or cross-validation, not repeated test looks), but a single project owner making every downstream decision while repeatedly seeing the same test set's numbers is a real, if soft, form of researcher degrees of freedom that a multi-analyst team with a held-out final test set would not have. Worth naming explicitly rather than leaving implicit.

## 3. NEW: CLV model forecast validation (real holdout, never checked before)

Step 12 validated BG/NBD and Gamma-Gamma's *assumptions* (independence, the frequency=0 instability) but never checked their *forecasts* against real subsequent behaviour. This is possible without sacrificing the horizon, by a genuine coincidence: the deployed model's fit cutoff (2011-06-09) plus its 183-day horizon lands exactly on 2011-12-09 — the last real day of transaction data in the dataset. The entire remainder of the dataset is untouched, genuine holdout.

**Methodology check, not assumed**: the forward window here (`invoice_ts >= cutoff+1day`, `< cutoff+1day+183days`) is copied from `sql/build_features.sql`'s own churn-label query. Using that SAME raw `invoice_type='SALE'` existence check (no merchandise filtering), this script finds exactly **1,838** customers with no forward purchase — bit-exact against the project's already-published churned count. (An earlier version of this check used a slightly different date boundary and produced 1,832; caught and fixed before trusting anything downstream of it.)

**A second, smaller definitional gap found in the process**: BG/NBD's own transaction definition (`src/models/clv.py::load_customer_transactions`) additionally requires a qualifying merchandise line (`item_type='PRODUCT'`, `quantity>0`, `unit_price>0`) — the churn label's raw SQL does not. Applying BG/NBD's own definition to the same window finds **1,842** zero-purchase customers, 4 more than the churn label's count. All 4 are real customers whose only forward SALE invoice contained solely a zero-priced line (e.g. a promotional item) — correctly excluded from BG/NBD's notion of a genuine purchase, but counted as "retained" by the churn label. Not a bug in either definition, but a real, previously-undocumented inconsistency: the churn label and the CLV model do not use quite the same definition of "the customer bought something." The validation below uses BG/NBD's own definition, since that is what its forecast should logically be checked against.

### Frequency: BG/NBD's `expected_purchases`

**Pearson r = 0.846** between predicted and real purchase counts over 4,323 customers (MAE 1.145 purchases). Well-calibrated in aggregate across every decile — no systematic over- or under-prediction band:

| decile | n | mean_predicted | mean_actual |
| --- | --- | --- | --- |
| 0 | 435 | 0.347 | 0.347 |
| 1 | 430 | 0.443 | 0.44 |
| 2 | 432 | 0.566 | 0.484 |
| 3 | 434 | 0.742 | 0.76 |
| 4 | 431 | 0.939 | 0.896 |
| 5 | 432 | 1.208 | 1.222 |
| 6 | 432 | 1.524 | 1.535 |
| 7 | 432 | 1.998 | 1.87 |
| 8 | 432 | 2.921 | 2.801 |
| 9 | 433 | 6.792 | 7.309 |

![CLV frequency forecast validation](reports/figures/clv_forecast_validation_frequency.png)

### Monetary: Gamma-Gamma's `expected_value_per_purchase`

| value_source | n_customers_with_a_future_purchase | pearson_r | mean_predicted | mean_actual |
| --- | --- | --- | --- | --- |
| gamma_gamma_conditional_expectation | 2119 | 0.8352 | 446.49 | 462.38 |
| own_observed_transaction (Gamma-Gamma unstable at frequency=0) | 362 | 0.0059 | 428.48 | 828.74 |

![CLV monetary forecast validation](reports/figures/clv_forecast_validation_monetary.png)

**A real, previously-undocumented limitation, found by this check**: Gamma-Gamma's own conditional expectation (repeat customers) is well-calibrated — r≈0.84, predicted and actual means within 4% of each other. The one-time-buyer FALLBACK (Step 12's fix for Gamma-Gamma's frequency=0 instability: use the customer's own single observed transaction value) has essentially **zero** correlation with what they actually spend on their next purchase, and underestimates it by roughly half. This does not mean Step 12's fix was wrong — the alternative it replaced was a provably worse, sometimes-negative estimate — but it means the ~30% of customers on the fallback path have a CLV figure that is a defensible point estimate, not an accurate individual forecast. This directly affects `retention_priority_score` for that subgroup and should be disclosed alongside any operational use of the ranked list.

## 4. Statistical rigor — limitations documented honestly

- **Single train/test split, no confidence interval.** Every headline metric (ROC-AUC 0.8115, etc.) comes from one particular 80/20 split. Its sensitivity to the random seed was never quantified (e.g. via repeated splits or bootstrap); a skeptical reviewer should treat the third decimal place of any reported metric as noise, not precision.
- **Multiple-testing correction scope.** Step 5's Bonferroni correction covers the 10 numeric Mann-Whitney tests; the categorical (chi-square, country) test was run and reported separately, uncorrected — consistent with treating it as one distinct hypothesis, but worth being explicit that it wasn't folded into the same correction family.
- **The business-cost framework's dollar figures are scenario outputs, not measurements** (already stated in Step 10, reaffirmed here): `contact_cost` and `retention_success_rate` are stated assumptions; only `value_per_customer` is measured. The reported net-value numbers are correct GIVEN the assumptions, not a business forecast.

## 5. LLM analyst layer — a brief robustness note

Not a full red-team exercise (out of scope here), but worth stating the actual attack surface: every tool is read-only, takes a small bounded set of typed arguments (a customer ID, an integer count, an enum), and executes no arbitrary code or free-text SQL. A successful prompt injection could at most cause an incorrect tool CALL (e.g. the wrong customer ID) — every tool's own input validation and the `{"error": ...}` contract (Step 21) still apply, and no tool can be made to mutate data, run arbitrary queries, or exfiltrate anything beyond what `/predict` and the dashboard already expose to anyone.

## Overall verdict

No new data-leakage or correctness bug was found in the core churn-modelling pipeline. One new empirical validation (the CLV forecast check above) was performed and is a genuine, positive result for the majority of the population (repeat customers) alongside one honestly surfaced limitation (the one-time-buyer value fallback). Every previously-documented limitation across the project was re-checked and reaffirmed as accurately described — none were found to be understated.
