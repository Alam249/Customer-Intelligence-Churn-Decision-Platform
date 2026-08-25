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
| 6 | Feature engineering | ⬜ Not started |
| 7 | Baseline model (Logistic Regression) | ⬜ Not started |
| 8–13 | Advanced models, tuning, calibration, SHAP, CLV, segmentation | ⬜ Not started |
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
