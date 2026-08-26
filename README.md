# Customer Intelligence & Churn Decision Platform

An end-to-end data science system that predicts customer churn, estimates customer
value, explains individual predictions, and ranks customers for retention action —
exposed through a REST API and an analyst dashboard.

> **Status:** in development. This README documents what is actually implemented.
> Sections are added as each component is completed. No metrics are reported until
> they have been produced from the real data.

---

## Build status

| Step | Component | Status |
| --- | --- | --- |
| 1 | Project architecture, config, logging | ✅ Complete |
| 2 | Dataset selection & profiling | ✅ Complete |
| 3 | PostgreSQL relational pipeline | ✅ Complete |
| 4 | Data quality & validation | ✅ Complete |
| 5 | Exploratory data analysis | ✅ Complete |
| 6 | Feature engineering | ✅ Complete |
| 7 | Baseline model (Logistic Regression) | ✅ Complete |
| 8 | Advanced model comparison (Random Forest, XGBoost) | ✅ Complete |
| 9 | Hyperparameter optimization (Optuna) | ✅ Complete |
| 10 | Probability calibration & business threshold | ✅ Complete |
| 11 | Explainable AI (SHAP) | ✅ Complete |
| 12 | Customer Lifetime Value & Retention Priority | ✅ Complete |
| 13 | Customer segmentation | ✅ Complete |
| 14–17 | FastAPI, Streamlit, MLflow, Docker | ⬜ Not started |
| 18–21 | Testing, monitoring, uplift, LLM layer | ⬜ Not started |

---

## Business problem

For a non-subscription retailer, customers never formally cancel — they simply stop
buying. That makes churn *invisible* until revenue has already been lost. This project
turns silent lapse into a measurable, predictable, and actionable signal:

1. **Predict** which active customers will stop purchasing in the next six months.
2. **Value** each customer so that retention spend is not wasted on low-value accounts.
3. **Explain** each prediction so an analyst can act on it rather than trust a black box.
4. **Prioritise** a ranked call list combining risk and value.

---

## Dataset

**UCI Online Retail II** — real transaction records from a UK-based online gift retailer.

| Property | Value |
| --- | --- |
| Source | [UCI Machine Learning Repository, ID 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) — free for public portfolio use |
| Grain | One row per product line item on an invoice |
| Rows | 1,067,371 |
| Period | 2009-12-01 to 2011-12-09 |
| Identified customers | 5,942 |
| Invoices | 53,628 |
| Products (stock codes) | 5,305 |
| Countries | 43 |

The dataset has **no churn column**. Churn is a *derived* label: a customer active in
the year before a fixed cutoff date who makes no purchase in the following 183 days.
The cutoff and horizon live in [config/config.yaml](config/config.yaml) so the
definition is explicit and reproducible rather than buried in code.

### Getting the data

The raw CSV belongs at `data/raw/online_retail_II.csv` (gitignored — it is not
committed).

```bash
# Download from the UCI repository
curl -L -o data/external/online_retail_II.zip \
  https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip -p data/external/online_retail_II.zip > data/raw/online_retail_II.xlsx
# then convert to CSV, or download the CSV export directly from Kaggle
```

---

## Data pipeline

```text
data/raw/online_retail_II.csv          1,067,371 denormalised line items
        │
        │  src/data/build_relational.py   — split into 5 normalised tables
        ▼
data/interim/*.csv                     countries, customers, products,
        │                              invoices, invoice_lines
        │  sql/schema.sql + sql/load_data.sql   — DDL, constraints, \copy, indexes
        ▼
PostgreSQL 16  ·  customer_intelligence
        │
        │  sql/build_features.sql       — labels + 22 features for one cutoff
        ▼
customer_features ⋈ churn_labels
        │
        │  sql/validation.sql           — 20 integrity & leakage assertions
        ▼
data/processed/customer_features_2011-06-09_h183.parquet
                                       4,323 customers × 25 columns
```

The export filename records the label definition, and the export query filters on
both `cutoff_date` and `horizon_days` — `churn_labels` is designed to hold several
definitions at once, so an unfiltered join would silently concatenate contradictory
targets for the same customer.

One command rebuilds the whole thing (**29.5 s** end to end on an M-series Mac):

```bash
python scripts/run_pipeline.py
```

### Entity-relationship diagram

```text
┌──────────────────┐
│    countries     │
├──────────────────┤
│ PK country_id    │──────────────┐
│    country_name  │              │
└──────────────────┘              │
         ▲                        │
         │ FK primary_country_id  │ FK country_id
         │                        │
┌──────────────────┐      ┌───────┴───────────────┐      ┌────────────────────┐
│    customers     │      │       invoices        │      │      products      │
├──────────────────┤      ├───────────────────────┤      ├────────────────────┤
│ PK customer_id   │◄─────│ PK invoice_no         │      │ PK stock_code      │
│ FK primary_      │  FK  │ FK customer_id  (NULL)│      │    description     │
│    country_id    │      │ FK country_id         │      │    item_type       │
└──────────────────┘      │    invoice_ts         │      └────────────────────┘
         ▲                │    invoice_type       │                 ▲
         │                └───────────────────────┘                 │
         │                          ▲                               │
         │                          │ FK invoice_no                 │ FK stock_code
         │                ┌─────────┴─────────────────────┐         │
         │                │        invoice_lines          │─────────┘
         │                ├───────────────────────────────┤
         │                │ PK line_id        (surrogate) │
         │                │    quantity, unit_price       │
         │                │    line_revenue   (GENERATED) │
         │                └───────────────────────────────┘
         │
         ├───────────────────────────────┬──────────────────────────────┐
         │ FK customer_id                │ FK customer_id               │
┌────────┴──────────────────┐   ┌────────┴─────────────────┐            │
│      churn_labels         │   │    customer_features     │            │
├───────────────────────────┤   ├──────────────────────────┤            │
│ PK (customer_id,          │   │ PK (customer_id,         │            │
│     cutoff_date,          │◄──┤     cutoff_date)         │            │
│     horizon_days)         │   │    22 feature columns    │            │
│    is_churned             │   └──────────────────────────┘            │
└───────────────────────────┘                                           │
                                                    derived tables ─────┘
```

**Key design decisions**, each verified against the raw data rather than assumed:

| Decision | Evidence |
| --- | --- |
| Country lives on `invoices`, not `customers` | 13 of 5,942 customers transact from more than one country |
| `invoice_lines` needs a surrogate PK | `(invoice_no, stock_code)` has 45,947 duplicate pairs |
| `products.description` is the *modal* description | 1,232 stock codes carry up to 9 different descriptions |
| `stock_code` is upper-cased before use as a PK | 173 codes differ from another only by case |
| `invoice_ts` = `MIN(InvoiceDate)` per invoice | 83 invoices span multiple timestamps (median spread 60 s) |
| Returns are `invoice_type = 'CREDIT'`, not a separate table | A `returns` table would duplicate identical columns |
| No `subscriptions` / `payments` / `support` tables | That data does not exist in this source and will not be invented |

### The churn label

The source has no churn column, so the target is derived and parameterised in
[config/config.yaml](config/config.yaml):

> A customer who bought merchandise in the **365 days before 2011-06-09** and made
> **zero purchases in the 183 days after it** is labelled churned.

| | |
| --- | --- |
| Eligible customers | **4,323** |
| Churned | **1,838 (42.52%)** |
| Retained | 2,485 (57.48%) |

The composite key on `churn_labels` lets several cutoff/horizon definitions coexist,
so the sensitivity of results to the label choice can be tested without overwriting
anything. The label is *highly* sensitive to the horizon, which is exactly why it is
a config parameter rather than a hard-coded constant:

| Cutoff | Horizon | Eligible | Churn rate |
| --- | --- | --- | --- |
| 2011-06-09 | 183 d | 4,323 | **42.52%** |
| 2011-03-09 | 91 d | 4,273 | 62.88% |

The 183-day definition is the project default: a 91-day window labels customers who
simply buy quarterly as churned, inflating the positive class to a degree that
reflects purchase cadence more than genuine attrition.

### Leakage control

Every feature CTE in `build_features.sql` filters `invoice_ts < cutoff + 1 day`. The
label CTE is the only code that reads post-cutoff rows, and it reads nothing but the
*existence* of a later sale. Four assertions in `sql/validation.sql` verify this holds:

```text
11 | PASS | LEAKAGE: no feature row has negative recency (purchase after cutoff)
12 | PASS | LEAKAGE: no feature row has recency > tenure
13 | PASS | LEAKAGE: every eligible customer purchased within the lookback window
14 | PASS | LEAKAGE: labels and features cover exactly the same customers
```

19 of 20 checks pass outright. Check 9 reports `KNOWN`: invoice `C496350` carries one
positive-quantity line on a credit note — a genuine source defect, documented with a
tolerance rather than silently patched.

### Engineered features (22)

Computed in SQL, strictly from the observation window:

| Group | Features |
| --- | --- |
| RFM core | `recency_days`, `frequency`, `monetary_total`, `monetary_avg_order` |
| Tenure & cadence | `tenure_days`, `active_days`, `avg_interpurchase_days`, `std_interpurchase_days`, `purchase_rate_per_month` |
| Basket | `total_items`, `avg_items_per_order`, `distinct_products`, `avg_unit_price` |
| Returns | `return_invoices`, `return_value`, `return_rate` |
| Recent activity | `orders_last_30d`, `orders_last_90d`, `spend_last_90d`, `spend_ratio_90d` |
| Context | `country_name`, `is_uk` |

Monetary features count **merchandise only** — postage, bank charges, discounts,
samples, vouchers and adjustments are classified via `products.item_type` and
excluded, without deleting the rows from the database.

### Database setup

```bash
# macOS
brew install postgresql@16
brew services start postgresql@16
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# Create the role and database
psql -d postgres -c "CREATE ROLE ci_user WITH LOGIN PASSWORD '<your-password>';"
psql -d postgres -c "CREATE DATABASE customer_intelligence OWNER ci_user;"

# Record the credentials (this file is gitignored)
cp .env.example .env && $EDITOR .env

python scripts/run_pipeline.py
```

Credentials reach `psql` through libpq environment variables via
`src.config.get_psql_env()`, so the password never appears in a process argument
list or in an error message.

Useful flags:

```bash
python scripts/run_pipeline.py --skip-build          # reuse data/interim CSVs
python scripts/run_pipeline.py --skip-load           # features only, no reload
python scripts/run_pipeline.py --cutoff 2011-03-09 --horizon 91   # alternate label
```

### Verifying the pipeline

```bash
psql -d customer_intelligence -c "
  SELECT cutoff_date, count(*) AS customers,
         round(100.0*avg(is_churned::int), 2) AS churn_rate_pct
  FROM churn_labels GROUP BY 1;"
```

---

## Data quality and validation

`sql/validation.sql` (Step 3) checks *pipeline integrity* — referential integrity,
row-count reconciliation, and leakage. **Step 4 is a separate, statistical pass**:
distributions, outliers, class balance, and feature-target correlation, run on both
the raw CSV and the model-ready feature table.

```bash
python scripts/run_data_quality.py
```

This writes [reports/data_quality_report.md](reports/data_quality_report.md) — every
number in it is measured from the real data, not asserted. The checks themselves live
in [src/data/quality.py](src/data/quality.py) as small, reusable functions (missing-value
report, duplicate report, IQR outlier report, target-correlation leakage screen, etc.)
rather than one-off notebook code, so they can be reused in Step 5's EDA and asserted
on directly in Step 18's test suite.

**Nothing is dropped without justification.** Two examples of findings that were
investigated rather than assumed:

- **34,335 exact duplicate raw line items.** Inspection showed these are the same
  product logged as separate identical lines on one invoice (e.g. invoice `489517`,
  stock code `21912` appears 3 times identically) — a till/EPOS pattern, not a
  data-entry error. **Kept** — deduplicating would understate revenue.
- **`return_rate` > 1.0 for 9 customers** (returns exceed in-window purchases, up to
  4.27×) — genuine (returning stock bought before the lookback window began), not a
  bug. **Capped at 1.0** for modelling stability; the original value is preserved in
  `return_rate_raw` rather than discarded.

Cleaning is a separate, explicit step from detection (`clean_customer_features` in
`src/data/quality.py`) and never touches the original file:

```text
data/processed/customer_features_2011-06-09_h183.parquet             (untouched, 25 cols)
data/processed/customer_features_2011-06-09_h183_validated.parquet   (+3 cols: flags + raw return rate)
```

| Finding | Severity | Treatment |
| --- | --- | --- |
| 34,335 duplicate raw line items | Low | Keep — legitimate repeated EPOS entries |
| 22.77% of raw rows have no Customer ID | Expected | Keep in DB; excluded from features by design |
| Ambiguous country labels (`Unspecified`, `RSA`, ...) | Low | Keep; `is_uk` is the split that matters |
| Right-skewed monetary/frequency features (resellers) | Medium | Keep; log-transform for linear models (Step 7) |
| `return_rate` > 1 for 9 customers | Low | Capped at 1.0; raw value kept in `return_rate_raw` |
| Null interpurchase-gap for low-frequency customers | Expected | Kept null + `*_is_missing` flag added |
| `recency_days` correlates 0.39 with the target | Watch | Not leakage (pre-cutoff only); flagged for SHAP (Step 11) |
| Mild class imbalance (42.5% / 57.5%) | Low | `class_weight='balanced'`; no resampling needed |

---

## Exploratory data analysis

[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) — the narrative EDA. All plotting
and statistical-test logic lives in [src/eda.py](src/eda.py) (colorblind-safe,
palette-validated chart functions plus Mann-Whitney and chi-square helpers), so the
notebook only calls into it and tells the analytical story; the dashboard (Step 15)
will reuse the same functions rather than duplicating chart code.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

**Key findings** (full detail and evidence in the notebook):

- Every one of 10 numeric features tested differs significantly between churned and
  retained customers (Mann-Whitney, p < 0.001 after Bonferroni correction).
- **Recency dominates**: median 59 days (retained) vs. 196 days (churned); churn
  climbs from 16% to 69% across recency quintiles. Verified as legitimate, not
  leakage — pre-cutoff only, per the four SQL assertions in Step 3.
- **Monetary value and churn move in clean, opposite gradients**: 69.1% churn in
  the lowest-spend quintile vs. 11.9% in the highest — exactly why Step 12 combines
  value and risk instead of ranking on churn probability alone.
- **Recent (90-day) activity separates customers more sharply than lifetime totals**:
  58.3% churn with no order in the last 90 days pre-cutoff vs. 23.7% with at least one.
- **Product-catalogue breadth is protective**: 63.6% churn (bottom quintile of
  distinct products bought) down to 15.6% (top quintile).
- **Tenure's relationship with churn is non-monotonic** — highest in the
  second-shortest tenure bin, not the shortest — consistent with new customers not
  yet having had a fair chance to re-order inside the observation window.
- Geography is a weak signal: `is_uk` alone is not significant (p=0.44); the full
  country breakdown is (p=0.007), but 91.5% of customers are UK-based and most
  non-UK countries have samples too small (<25 customers) to act on.
- `frequency` and `active_days` carry near-duplicate information (r=0.961) —
  relevant for the linear baseline (Step 7), less so for tree-based models (Step 8).

All figures are saved to [reports/figures/](reports/figures/):

<p>
  <img src="reports/figures/target_balance.png" width="420" alt="Churn label distribution">
  <img src="reports/figures/binned_monetary_total_churn_rate.png" width="420" alt="Churn rate by monetary total quintile">
</p>

---

## Feature engineering

```bash
python scripts/run_feature_engineering.py
```

Reads the Step 4 validated feature table (4,323 customers) and produces
`data/processed/train.parquet` (3,458), `data/processed/test.parquet` (865), the
fitted transformer `models/feature_engineer.joblib`, and
[reports/feature_engineering_report.md](reports/feature_engineering_report.md) — a
full dictionary of every feature reaching the model (formula, business meaning,
churn hypothesis, leakage risk), generated from
[src/features/catalog.py](src/features/catalog.py).

**Split strategy: stratified random, not time-based.** The feature table is a
single cross-sectional snapshot — one row per customer at one fixed cutoff — so
there is no per-row time axis to split on; a genuine time-based split would need
multiple snapshot cutoffs, a larger exercise reserved for future drift work
(Step 19). The split is stratified on `is_churned` (train 42.51% churned, test
42.54% — both within 0.1pp of the full dataset's 42.52%).

**New features** (on top of the 22 SQL features from Step 3 and the 2 quality
flags from Step 4), built by [`src/features/engineer.py`](src/features/engineer.py)'s
`CustomerFeatureEngineer`:

| Feature | Formula | Why |
| --- | --- | --- |
| `spend_per_tenure_month` | `monetary_total / (tenure_days / 30.44)` | Spend velocity, not just accumulated total |
| `orders_ratio_90d` | `orders_last_90d / frequency` | Scale-free trend signal (complements `spend_ratio_90d`) |
| `products_per_order` | `distinct_products / frequency` | Catalogue breadth per transaction |
| `purchase_regularity_cv` | `std_interpurchase_days / avg_interpurchase_days` | How predictable a customer's rhythm is — the closest honest substitute this dataset has for a "contract risk" signal (no contract data exists) |
| `rfm_score` | R/F/M quantile-scored 1–5, summed (3–15) | Classic composite score for reporting/segmentation |
| `is_high_value` | `monetary_total >= 75th pct(train)` | Business-readable top-quartile flag |

**Leakage discipline, proven not just claimed.** `rfm_score` and `is_high_value`
need a quantile threshold learned from data — exactly the kind of feature that
leaks if mishandled. `CustomerFeatureEngineer.fit()` is called on the training
split only; the test split reuses those thresholds rather than recomputing its
own. Verified concretely: the test split's own 75th percentile of
`monetary_total` is 2,487.30 (different from the 2,189.00 learned on train) — if
`is_high_value` had been (incorrectly) computed from the test split itself,
25.1% of test customers would be flagged by construction. The actual flagged
share is **28.3%**, proving the training threshold was applied, not a refit one.

`rfm_score` alone produces a clean, monotonic churn gradient on the training
split — 79.6% churn at the lowest score (3) down to 2.8% at the highest (14) —
without needing any model at all.

---

## Baseline model — Logistic Regression

```bash
python scripts/run_baseline_model.py
```

The first model, and deliberately **not tuned** — it's the benchmark every later
model (Step 8's comparison, Step 9's tuned candidate) must beat to justify its
added complexity. `sklearn.Pipeline` + `ColumnTransformer`: median imputation +
Yeo-Johnson power transform + standardisation for numeric features, passthrough
for booleans, `LogisticRegression(class_weight='balanced')`. All fit on the
training split only.

| Metric | Train | Test |
| --- | --- | --- |
| Accuracy | 0.7131 | 0.7214 |
| Precision | 0.6334 | 0.6494 |
| Recall | 0.7721 | 0.7500 |
| F1 | 0.6959 | 0.6961 |
| ROC-AUC | 0.7904 | **0.8018** |
| PR-AUC | 0.7118 | **0.6966** |

Train and test scores are close — no meaningful overfitting. **Accuracy alone
would be misleading here**: a trivial "always predict retained" model scores
57.5% accuracy while catching zero churners, which is worthless for a retention
program. ROC-AUC and especially PR-AUC (test base rate 42.5%) are reported for
exactly that reason.

**A real finding, reported rather than hidden**: for 13 of the 27 features, the
fitted coefficient's sign disagrees with that feature's own univariate
correlation with churn — including `rfm_score`, which has the single strongest
univariate correlation of any feature (-0.474) yet nearly vanishes in the
multivariate fit. The design matrix's condition number is 145.9 (>30 signals
real multicollinearity). This doesn't hurt the model's predictions or ranking —
only the interpretability of individual coefficients — and is diagnosed in full,
feature-by-feature, in
[reports/baseline_model_report.md](reports/baseline_model_report.md). Step 9's
regularisation search and Step 8's tree-based models (not sensitive to
collinearity) are the paths to a model whose coefficients/importances can be
read at face value.

Excluded from this linear model specifically (each for a measured reason —
Step 8's trees may reuse them): `country_name`, `active_days`, the three RFM
sub-scores, `return_rate_raw`, `orders_ratio_90d`.

<p>
  <img src="reports/figures/roc_curve.png" width="280" alt="ROC curve">
  <img src="reports/figures/pr_curve.png" width="280" alt="Precision-Recall curve">
  <img src="reports/figures/confusion_matrix.png" width="280" alt="Confusion matrix">
</p>

---

## Model comparison — Random Forest and XGBoost vs. the baseline

```bash
python scripts/run_model_comparison.py
```

| Model | Test ROC-AUC | Test PR-AUC | Overfit gap (train − test AUC) | Train time |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.8018** | **0.6966** | -0.011 | (Step 7) |
| Random Forest | 0.7993 | 0.6954 | 0.201 | 0.41s |
| XGBoost | 0.7679 | 0.6780 | 0.232 | 0.50s |

**The untuned Logistic Regression baseline wins.** Real result, not a bug: with
only 3,458 training rows, both untuned tree ensembles show substantial
overfitting (see the overfit-gap column) that the much lower-capacity linear
model doesn't. XGBoost's top-5 feature importances even include
`country_Denmark` — a country with only 6 training customers, 0% of them
churned, i.e. the model memorized a subgroup this small rather than learning a
real geographic effect. This is the concrete evidence behind "the most complex
model is not automatically the best," and it's exactly the scenario Step 9's
hyperparameter search (constraining depth, adding regularization) targets —
not evidence trees are the wrong model family here.

Both tree models use the **full 34-column feature set**, including everything
Step 7 excluded for the linear model on collinearity grounds — trees aren't
sensitive to that problem. Random Forest's importances spread fairly evenly
across ~15 features; XGBoost's concentrate heavily on `rfm_score` alone
(importance 0.27, more than double the next feature) — a first hint of the two
algorithms' very different sensitivity to correlated inputs, worth revisiting
with SHAP in Step 11.

A second boosting library (LightGBM/CatBoost) was deliberately not added:
Random Forest and XGBoost already span bagging vs. boosting, and a second
booster would mostly repeat XGBoost's story on a dataset this size rather than
add a genuinely different comparison point.

<p>
  <img src="reports/figures/roc_comparison.png" width="360" alt="ROC curve comparison across three models">
  <img src="reports/figures/xgb_feature_importance.png" width="420" alt="XGBoost feature importance">
</p>

Full discussion — interpretability trade-offs, business cost of each model
choice, and reasoning for the Step 9 tuning candidate — in
[reports/model_comparison_report.md](reports/model_comparison_report.md).

---

## Hyperparameter optimization — XGBoost

```bash
python scripts/run_hyperparameter_tuning.py
```

**Candidate: XGBoost**, chosen because Step 8 diagnosed it (not Random Forest)
as showing the largest overfitting gap (0.232) with the richest set of
regularization controls to fix it directly. Search: **Optuna** (TPE sampler),
50 trials, stratified 5-fold CV on the training split only — preferred over
RandomizedSearchCV because the 9-parameter, mostly-continuous space is exactly
where a sequential, model-guided sampler out-samples uniform random draws.
Optimized for **PR-AUC (average precision)**, not ROC-AUC: under this mild
imbalance, ROC-AUC can improve entirely via the tail nobody acts on, while
PR-AUC tracks the positive-class ranking that actually drives Step 12's
retention-priority list.

| Model | Test ROC-AUC | Test PR-AUC | Overfit gap |
| --- | --- | --- | --- |
| XGBoost (untuned, Step 8) | 0.7679 | 0.6780 | 0.232 |
| **XGBoost (tuned)** | **0.8091** | **0.6998** | **-0.008** |
| Logistic Regression (Step 7/8) | 0.8018 | 0.6966 | -0.011 |

**Tuning genuinely fixed the overfitting** (gap closed by 0.24, essentially to
zero) **and the tuned model now beats the linear baseline** by +0.0073
ROC-AUC — achieved through regularization (`max_depth=3`, strong L1/L2,
`gamma=3.95`), not raw capacity. The search converged by trial 4 of 50 and
never meaningfully improved after — the trial budget was more than sufficient.
The test set was touched exactly once, after the search completed, never
during the 50-trial CV loop.

<p>
  <img src="reports/figures/optuna_history.png" width="380" alt="Hyperparameter search convergence">
  <img src="reports/figures/roc_comparison_tuning.png" width="380" alt="Tuned vs untuned XGBoost ROC comparison">
</p>

Full search space, top trials, and best parameters in
[reports/hyperparameter_tuning_report.md](reports/hyperparameter_tuning_report.md);
all 50 trial results in `reports/optuna_trials.csv`.

---

## Probability calibration and business threshold

```bash
python scripts/run_calibration_and_threshold.py
```

Uses the Step 9 tuned XGBoost model — but checks something ROC-AUC/PR-AUC
never test: **are its predicted probabilities trustworthy as probabilities?**
`scale_pos_weight` (used to correct class imbalance) is known to distort this
even when it helps ranking metrics.

| Probabilities | Brier score |
| --- | --- |
| Raw (tuned XGBoost) | 0.1804 |
| Calibrated (sigmoid) | 0.1755 |
| **Calibrated (isotonic)** | **0.1746** |

Isotonic calibration measurably improves the Brier score, fit via 5-fold CV on
the training split only (test never touched during fitting) — **adopted**,
saved as `models/final_churn_model.joblib`. The reliability diagram confirms
why: the raw model is visibly underconfident in the mid-probability range.

**Business-cost framework** (a demonstration, explicitly labeled): Online
Retail II has no record of any real retention campaign, so `contact_cost` and
`retention_success_rate` are stated, hypothetical assumptions — only
`value_per_customer` (€899.54, median training `monetary_total`) is measured.

| Scenario | Contact cost | Success rate | Optimal threshold |
| --- | --- | --- | --- |
| Cheap, low-touch | €2 | 12% | 0.08 |
| Primary (moderate offer) | €15 | 25% | 0.08 |
| Expensive, high-touch | €120 | 20% | **0.36** |

The threshold genuinely moves with the assumptions — not a token gesture: an
expensive, uncertain-payoff intervention pushes the model to be far more
selective. **Caveat reported alongside the recommendation, not hidden**: the
primary scenario's "optimal" threshold (0.08) flags 84% of test customers —
mathematically correct given the stated costs, but closer to a mass campaign
than a targeted list; a real team's capacity constraint isn't part of this
simple framework, and a fixed contact-list size is often the more usable
operational answer.

<p>
  <img src="reports/figures/calibration_curve.png" width="320" alt="Calibration reliability diagram">
  <img src="reports/figures/threshold_curves.png" width="420" alt="Precision and recall vs decision threshold">
</p>

Full detail — threshold-performance table, cost-framework mechanics, and the
uplift-modeling caveat this framework doesn't capture (Step 20's territory) —
in [reports/calibration_threshold_report.md](reports/calibration_threshold_report.md).

---

## Explainable AI — SHAP

```bash
python scripts/run_shap_explainability.py
```

**Which model SHAP explains, and why**: the final model (Step 10) is a
`CalibratedClassifierCV` — internally 5 cloned pipelines, one per calibration
fold — which SHAP's `TreeExplainer` cannot open directly. SHAP attribution is
computed on the pre-calibration Step 9 tuned XGBoost pipeline (calibration
rescales the output; it doesn't change what the trees split on); the
probability shown for every customer still comes from the calibrated final
model, matching what Step 14's API will return.

**Global**: `rfm_score` dominates (mean |SHAP| 0.31, ~3x the runner-up
`recency_days` at 0.11), with an almost perfectly monotonic dependence plot.
This directly **contrasts with Step 7's Logistic Regression**, where
`rfm_score` had the strongest univariate correlation of any feature yet its
coefficient nearly vanished — absorbed by its own raw ingredients still in the
linear model. Trees don't have that problem: a concrete, measured illustration
of why Step 8 compared multiple model families instead of trusting one's
feature ranking.

**Local**: a reusable `explain_customer(customer_id, ...)` function (used
identically here and by the future Step 14 `/predict/explain` endpoint) —
narrative + a red/blue waterfall-style chart using the same churn-color
convention as every chart since Step 5:

> Customer 14822: 94.9% predicted churn probability (High risk). Top factors
> increasing risk: rfm_score = 3 (+0.35); recency_days = 302 (+0.19);
> monetary_total = 158 (+0.14)...

**What SHAP does and doesn't mean, stated explicitly**: values are in log-odds
space (not literal probability points), and — critically — **a high-impact
feature is not a cause of churn**. `recency_days` dominates partly because
it's mechanically close to how churn is *labelled*; that's expected, not
evidence of causation. Real causal claims need Step 20's experimental design,
not an explainability method applied to an observational model.

<p>
  <img src="reports/figures/shap_summary.png" width="380" alt="SHAP beeswarm summary plot">
  <img src="reports/figures/shap_local_14822.png" width="380" alt="Local SHAP explanation for a high-risk customer">
</p>

Full detail in [reports/shap_explainability_report.md](reports/shap_explainability_report.md).

> **Dependency note**: SHAP 0.49.1 (latest) cannot parse XGBoost ≥3.0's
> `base_score` serialization format ([shap#4202](https://github.com/shap/shap/issues/4202)).
> `xgboost` is pinned to `2.1.4` in requirements.txt; Steps 8-10 were
> regenerated under this version and produced byte-identical results.

---

## Customer Lifetime Value and Retention Priority

```bash
python scripts/run_clv_and_retention_priority.py
```

**Methodology: BG/NBD + Gamma-Gamma**, not a simpler average-spend proxy —
justified because Online Retail II genuinely has what this needs: a full
multi-year repeat-purchase transaction history per customer in a
non-contractual setting, the textbook use case these models were built for.
CLV is projected over the same 183-day horizon as the churn label, using each
customer's complete transaction history (not the churn model's 365-day
lookback window).

Checked, not assumed: Gamma-Gamma's independence assumption (correlation
between frequency and monetary value = 0.084, negligible — holds). **Found by
testing, not theory**: Gamma-Gamma's conditional-expectation formula is
unstable at `frequency=0` — verified it returns a *negative* "expected
profit" for one-time buyers — confirming why they're handled with their own
observed transaction value instead (32.5% of the population, so this isn't a
minor edge case).

**Retention Priority Score** = `churn_probability × CLV` — the expected
revenue at risk if nothing is done, deliberately the simplest defensible
formula rather than an arbitrarily "improved" one.

**Why churn probability alone is not enough — measured**: ranking the same
200-customer contact budget by churn probability alone vs. the combined
score:

| Ranking strategy | Total CLV-at-risk captured |
| --- | --- |
| Churn probability alone | €7,172 |
| **Retention priority score** | **€118,980** |

The two lists share **zero customers**. The highest churn-probability
customers are long-dormant, zero-repeat-purchase accounts worth €50-60 each —
the model is confident they're already gone, and there's little left to
protect. The top priority-ranked customer has only 37% churn probability but
an estimated €32,668 CLV.

**A limitation surfaced, not hidden**: the pure product formula lets extreme
CLV outliers dominate the ranking even at low risk (8 of the top 10 are "Low
risk / High value," not "High risk / High value") — mathematically correct,
but a customer at 1.6% churn probability is already essentially certain to
stay, so ranking them highly overstates how actionable they are. A real
deployment would filter to a minimum meaningful risk threshold first (e.g.
Step 10's cost-optimal threshold) before ranking by priority score.

<p>
  <img src="reports/figures/retention_quadrant.png" width="380" alt="Risk vs value segmentation quadrant">
  <img src="reports/figures/targeting_comparison.png" width="320" alt="Targeting strategy comparison">
</p>

Full detail, fitted model parameters, and the ranked list for all 4,323
customers in
[reports/clv_retention_priority_report.md](reports/clv_retention_priority_report.md)
and `reports/retention_priority_list.csv`.

---

## Customer segmentation

```bash
python scripts/run_customer_segmentation.py
```

K-Means on RFM + tenure + catalogue breadth + 90-day engagement + Step 12's
CLV estimate (Yeo-Johnson power-transformed and standardised — Euclidean
distance is scale-sensitive the same way Step 7's linear model was).
`rfm_score` is deliberately excluded to avoid double-counting the same signal
its own inputs already provide.

**K chosen by evidence, not convenience**: silhouette peaks at K=3 (0.363),
but K=4 (0.330) is used instead — a stated trade-off. At K=3 the customers
who've gone quiet form one cluster; at K=4 that splits into a **moderate-value,
still-reachable** group and a **near-total-loss, one-time-buyer** group with a
23-point churn-rate gap between them. Validated two ways: split-half stability
ARI=0.957 (highly reproducible across resamples) and K-Means vs. Hierarchical
agreement ARI=0.729 (not a K-Means-specific artifact).

| Segment | Churn rate | Median CLV | Profile |
| --- | --- | --- | --- |
| **Champions** | 11.4% | €1,139 | Lowest recency, highest frequency/value/tenure/breadth |
| Declining / Moderate value | 47.5% | €310 | Recency climbing, still reachable |
| New / Developing | 41.7% | €322 | Short tenure, most history is recent by definition |
| **Lost / One-time buyers** | 70.5% | €105 | Frequency ≈1, longest recency, hardest to influence |

**Does segmentation add value beyond the supervised model?** Cross-tabulated
against Step 12's risk/value quadrant: **ARI = 0.32** (low-to-moderate) — the
clusters are *not* simply re-deriving that simpler 2×2 split. "Declining" and
"New" customers, in particular, spread across three of the four quadrants
each, showing segmentation surfaces tenure/engagement structure the risk ×
value view alone collapses away.

<p>
  <img src="reports/figures/cluster_profile_heatmap.png" width="380" alt="Cluster profile heatmap">
  <img src="reports/figures/cluster_churn_value.png" width="420" alt="Churn rate and CLV by cluster">
</p>

Full detail in [reports/segmentation_report.md](reports/segmentation_report.md);
segmented list for all 4,323 customers in `reports/customer_segments.csv`.

---

## Repository structure

```text
customer-intelligence-platform/
├── api/                  FastAPI prediction service (routers, Pydantic schemas)
├── config/               config.yaml — all project decisions in one reviewable file
├── dashboard/            Streamlit analyst dashboard
├── data/
│   ├── raw/              Immutable source data. Never written to by any script.
│   ├── interim/          Intermediate artefacts (relational CSVs for DB loading)
│   ├── processed/        Model-ready feature tables and train/test splits
│   └── external/         Third-party reference data and downloads
├── logs/                 Rotating pipeline logs (gitignored)
├── models/               Serialised models and preprocessing pipelines (gitignored)
├── notebooks/            Exploration and narrative analysis only — no library code
├── reports/
│   └── figures/          Saved visualisations used in the README and dashboard
├── scripts/              Runnable CLI entry points that orchestrate src/ modules
├── sql/                  schema.sql, load scripts, analytical feature queries
├── src/
│   ├── data/             Loading, DB I/O, raw → relational transformation
│   ├── features/         Feature engineering transformers
│   ├── models/           Training, tuning, persistence
│   ├── evaluation/       Metrics, plots, evaluation reports
│   ├── utils/            Logging and shared helpers
│   └── config.py         Config + path resolution + database URL construction
├── tests/                pytest suite
├── .env.example          Template for secrets — copy to .env (gitignored)
├── Makefile              Reproducible entry points
└── requirements.txt
```

### Why the structure is shaped this way

- **`src/` vs `notebooks/`** — every reusable function lives in `src/` and is imported
  by notebooks. Notebooks tell the story; they never *are* the implementation. This is
  the single biggest difference between a portfolio project and a pile of notebooks.
- **`data/` is layered and immutable at the source** — `raw/` is written once and read
  forever. Any transformation produces a new artefact in `interim/` or `processed/`, so
  every result is traceable back to the original file.
- **`scripts/` separate from `src/`** — `src/` is importable library code with no side
  effects; `scripts/` are thin CLI wrappers. That keeps the library testable and lets
  the API and dashboard reuse the exact same code paths as the training pipeline.
- **`sql/` as a first-class folder** — the feature layer is built in PostgreSQL, so the
  SQL is version-controlled and reviewable alongside the Python.
- **`config/` centralises decisions** — the churn cutoff, the seed, and the cleaning
  rules are choices a reviewer should be able to find and challenge in one file.
- **`api/` and `dashboard/` are separate deliverables** — they are containerised
  independently in the Docker step and depend only on saved model artefacts.

---

## Configuration and logging

Configuration is a single YAML file loaded by [src/config.py](src/config.py), which
also resolves every path relative to the repository root — so scripts behave the same
regardless of the working directory they are launched from.

```python
from src.config import CONFIG, PATHS, RANDOM_SEED
from src.utils.logging import get_logger

logger = get_logger(__name__)
raw = PATHS.data_raw / CONFIG["data"]["raw_file"]
```

**Secrets are never in config.** Database credentials are read from environment
variables via `.env` (see `.env.example`); `src.config.get_database_url()` builds the
connection string and fails with a clear message if a variable is missing.

Logging is configured once per process by `src.utils.logging.get_logger`, writing to
both stdout and a rotating file in `logs/`. `LOG_LEVEL` in the environment overrides
the config value, which is what containers and CI need.

---

## Installation

Requires Python 3.10+.

```bash
git clone <repository-url>
cd customer-intelligence-platform

make setup                  # creates .venv and installs requirements.txt
source .venv/bin/activate

cp .env.example .env        # then edit with your local PostgreSQL credentials
```

## Verifying the setup

```bash
make verify                 # or: python scripts/verify_setup.py
```

Expected output confirms the config loads, all expected directories exist, and the raw
dataset is present:

```text
INFO | Project: customer-intelligence-platform
INFO | All 6 expected directories present.
INFO | Raw dataset found: online_retail_II.csv (94.9 MB)
INFO | Setup verification passed.
```

---

## Technology stack

| Layer | Tools |
| --- | --- |
| Data manipulation | Python 3.10, pandas, NumPy |
| Storage & feature layer | PostgreSQL, SQL, SQLAlchemy |
| Modelling | scikit-learn *(XGBoost / LightGBM in later steps)* |
| Explainability | SHAP *(later step)* |
| Experiment tracking | MLflow *(later step)* |
| Serving | FastAPI, Uvicorn *(later step)* |
| Dashboard | Streamlit *(later step)* |
| Packaging | Docker, Docker Compose *(later step)* |
| Quality | pytest, Ruff, Black *(later step)* |

Dependencies are added to `requirements.txt` as each step is implemented, so the
environment is installable at every commit rather than only at the end.

---

## Licence & attribution

Dataset: Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D — licensed CC BY 4.0.
