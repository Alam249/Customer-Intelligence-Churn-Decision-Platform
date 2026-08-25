"""The feature dictionary for the churn model — one entry per feature that
reaches the model, whichever stage produced it (SQL, Step 4's quality flags,
or Step 6's ``CustomerFeatureEngineer``).

Kept as structured data, not prose, so `scripts/run_feature_engineering.py`
can render it into a Markdown table and later steps (SHAP narrative in Step 11,
the README in Step 22) can import the same source of truth instead of a second
hand-written copy drifting out of sync with the code.
"""

from __future__ import annotations

FEATURE_CATALOG: list[dict[str, str]] = [
    # --- RFM core (SQL, Step 3) ---
    {"name": "recency_days", "stage": "SQL (Step 3)", "category": "RFM",
     "formula": "cutoff_date - MAX(invoice_date)",
     "business_meaning": "Days since the customer's last purchase before the cutoff.",
     "churn_hypothesis": "Longer since last purchase -> more likely to have already left.",
     "leakage_risk": "None: computed strictly from invoices <= cutoff (asserted in sql/validation.sql)."},
    {"name": "frequency", "stage": "SQL (Step 3)", "category": "RFM",
     "formula": "COUNT(DISTINCT invoice_no) in the eligibility window",
     "business_meaning": "Number of separate orders placed.",
     "churn_hypothesis": "Fewer orders -> weaker relationship -> higher churn risk.",
     "leakage_risk": "None."},
    {"name": "monetary_total", "stage": "SQL (Step 3)", "category": "RFM",
     "formula": "SUM(quantity * unit_price) over merchandise lines",
     "business_meaning": "Total revenue from the customer in the observation window.",
     "churn_hypothesis": "Lower lifetime spend -> less invested in the relationship.",
     "leakage_risk": "None."},

    # --- Cadence / tenure (SQL, Step 3) ---
    {"name": "tenure_days", "stage": "SQL (Step 3)", "category": "Customer history",
     "formula": "cutoff_date - MIN(invoice_date)",
     "business_meaning": "How long the customer has been buying at all.",
     "churn_hypothesis": "Non-monotonic (EDA, Step 5): very short tenure inflates apparent churn "
                          "(not enough time to re-order yet); long tenure is protective.",
     "leakage_risk": "None."},
    {"name": "active_days", "stage": "SQL (Step 3)", "category": "Customer history",
     "formula": "COUNT(DISTINCT invoice_date)",
     "business_meaning": "Distinct calendar days with a purchase.",
     "churn_hypothesis": "More distinct purchase days -> more engaged.",
     "leakage_risk": "None. Correlated 0.961 with frequency (Step 4/5) — near-duplicate information."},
    {"name": "avg_interpurchase_days / std_interpurchase_days", "stage": "SQL (Step 3)",
     "category": "Customer history",
     "formula": "mean / stddev of gaps between consecutive purchase days",
     "business_meaning": "Typical time between orders, and how variable that rhythm is.",
     "churn_hypothesis": "Longer average gap -> lower-frequency relationship, weaker signal to retain.",
     "leakage_risk": "None. Null for customers with too few orders — flagged, not imputed (Step 4)."},
    {"name": "purchase_rate_per_month", "stage": "SQL (Step 3)", "category": "Customer history",
     "formula": "frequency / (tenure_days / 30.44)",
     "business_meaning": "Orders per month, normalising frequency by how long the customer has existed.",
     "churn_hypothesis": "A customer who buys often per month they've been active is lower-risk than "
                          "raw frequency alone would suggest for a long-tenure customer.",
     "leakage_risk": "None."},

    # --- Basket / catalogue (SQL, Step 3) ---
    {"name": "total_items / avg_items_per_order", "stage": "SQL (Step 3)", "category": "Monetary",
     "formula": "SUM(quantity) / AVG(quantity) per order",
     "business_meaning": "Basket size.",
     "churn_hypothesis": "Larger baskets suggest a more committed (or B2B/reseller) customer.",
     "leakage_risk": "None."},
    {"name": "distinct_products", "stage": "SQL (Step 3)", "category": "Behavioral",
     "formula": "COUNT(DISTINCT stock_code)",
     "business_meaning": "Catalogue breadth purchased — the closest proxy to \"service usage\" this "
                          "retail dataset supports.",
     "churn_hypothesis": "Confirmed in EDA (Step 5): 63.6% churn (bottom quintile) -> 15.6% (top) — "
                          "broader exploration of the catalogue is protective.",
     "leakage_risk": "None."},
    {"name": "avg_unit_price", "stage": "SQL (Step 3)", "category": "Monetary",
     "formula": "SUM(line_revenue) / SUM(quantity)",
     "business_meaning": "Quantity-weighted average price point purchased.",
     "churn_hypothesis": "Weak prior; included for completeness rather than a strong hypothesis.",
     "leakage_risk": "None."},

    # --- Returns (SQL, Step 3) ---
    {"name": "return_invoices / return_value / return_rate", "stage": "SQL (Step 3)",
     "category": "Behavioral",
     "formula": "count/value of CREDIT invoices; return_rate = return_value / monetary_total (capped at "
                "1.0 in the Step 4 validated table, original in return_rate_raw)",
     "business_meaning": "Return/refund behaviour — the closest proxy this dataset has to a "
                          "\"payment/service friction\" signal (there is no payment-method or failed-"
                          "payment data in Online Retail II).",
     "churn_hypothesis": "Counter-intuitively PROTECTIVE in EDA (Step 5): issuing a return requires an "
                          "active relationship, so zero-return customers are disproportionately "
                          "one-and-done buyers who were always likely to lapse. Read as an engagement "
                          "correlate, not a causal effect of returns.",
     "leakage_risk": "None — computed from pre-cutoff credit notes only."},

    # --- Recent activity (SQL, Step 3) ---
    {"name": "orders_last_30d / orders_last_90d / spend_last_90d / spend_ratio_90d",
     "stage": "SQL (Step 3)", "category": "Behavioral",
     "formula": "activity restricted to the 30/90 days before the cutoff; "
                "spend_ratio_90d = spend_last_90d / monetary_total",
     "business_meaning": "Trend / trajectory signal — is this customer's recent behaviour consistent "
                          "with their history, or have they gone quiet?",
     "churn_hypothesis": "Confirmed strongest behavioural gap in EDA after recency itself: 58.3% churn "
                          "with no order in the trailing quarter vs. 23.7% with at least one.",
     "leakage_risk": "None — all activity is <= cutoff. This is exactly the kind of feature that could "
                      "leak if a future engineer computed a `_next_N_days` version by mistake; "
                      "sql/validation.sql's leakage checks exist to catch precisely that."},

    # --- Context (SQL, Step 3) ---
    {"name": "country_name / is_uk", "stage": "SQL (Step 3)", "category": "Context",
     "formula": "modal shipping country per customer",
     "business_meaning": "Geography.",
     "churn_hypothesis": "Weak (EDA, Step 5): `is_uk` alone is not significant (p=0.44); the full "
                          "country breakdown is (p=0.007) but most non-UK countries have samples too "
                          "small (<25 customers) to act on.",
     "leakage_risk": "None."},

    # --- Data-quality flags (Step 4) ---
    {"name": "avg_interpurchase_days_is_missing / std_interpurchase_days_is_missing",
     "stage": "Data quality (Step 4)", "category": "Derived",
     "formula": "boolean indicator: was the corresponding gap feature null?",
     "business_meaning": "Distinguishes \"too few orders to compute a gap\" from \"gap is zero\" — "
                          "the two are behaviourally very different but would be conflated by any "
                          "form of mean/zero imputation.",
     "churn_hypothesis": "A customer with only one order (hence missing gap) is structurally different "
                          "from one with a very short, non-null gap.",
     "leakage_risk": "None — derived purely from the missingness pattern of pre-cutoff features."},

    # --- NEW in Step 6: row-wise derived features ---
    {"name": "spend_per_tenure_month", "stage": "Feature engineering (Step 6)", "category": "Monetary",
     "formula": "monetary_total / (MAX(tenure_days, 1) / 30.44)",
     "business_meaning": "Spend velocity — revenue per month of relationship, rather than an "
                          "accumulated total that rewards long tenure regardless of pace.",
     "churn_hypothesis": "A customer with high lifetime spend but low monthly velocity (i.e. spend "
                          "concentrated early, long tenure since) is a different risk profile than "
                          "raw `monetary_total` shows, and this feature should help separate them.",
     "leakage_risk": "None — a deterministic row-wise function of two existing pre-cutoff features. "
                      "Requires no fitted parameter, so it is identical whether computed before or "
                      "after the train/test split."},
    {"name": "orders_ratio_90d", "stage": "Feature engineering (Step 6)", "category": "Behavioral",
     "formula": "orders_last_90d / frequency",
     "business_meaning": "Share of a customer's ENTIRE order history that happened in the last "
                          "quarter — a scale-free version of the trend signal (spend_ratio_90d already "
                          "exists but is scale-sensitive to one large invoice).",
     "churn_hypothesis": "A customer whose orders are concentrated in the distant past even if their "
                          "spend_ratio_90d looks moderate is a stronger churn signal this feature "
                          "targets directly.",
     "leakage_risk": "None — row-wise, no fitted parameter."},
    {"name": "products_per_order", "stage": "Feature engineering (Step 6)", "category": "Behavioral",
     "formula": "distinct_products / frequency",
     "business_meaning": "Catalogue breadth PER TRANSACTION, separating a customer who buys many "
                          "different things across few orders from one who buys the same few items "
                          "repeatedly across many orders — `distinct_products` alone conflates them.",
     "churn_hypothesis": "Repeat-same-item buyers (low ratio, high frequency) may be a stable but "
                          "narrow relationship; explorers (high ratio) engage more broadly with the "
                          "catalogue, consistent with the protective effect of `distinct_products`.",
     "leakage_risk": "None — row-wise, no fitted parameter."},
    {"name": "purchase_regularity_cv", "stage": "Feature engineering (Step 6)", "category": "Derived",
     "formula": "std_interpurchase_days / avg_interpurchase_days (coefficient of variation)",
     "business_meaning": "How PREDICTABLE a customer's purchase timing is, independent of how "
                          "frequently they buy. This is the closest analogue this dataset supports to "
                          "a \"contract risk\" indicator (Online Retail II has no contract data at "
                          "all — this is an honest substitute, not a stand-in for one).",
     "churn_hypothesis": "An erratic purchase rhythm (high CV) is harder to predict and plausibly "
                          "riskier than a customer who reorders on a very regular clock, even at the "
                          "same average frequency.",
     "leakage_risk": "None — row-wise ratio of two existing pre-cutoff columns. Null wherever the "
                      "inputs are null (customers with <3 orders), inherited rather than newly created."},

    # --- NEW in Step 6: features requiring a train-fit threshold ---
    {"name": "recency_score / frequency_score / monetary_score / rfm_score",
     "stage": "Feature engineering (Step 6)", "category": "RFM",
     "formula": "each of R, F, M bucketed into up to 5 quantiles FIT ON THE TRAINING SPLIT ONLY, "
                "scored 1 (worst) to 5 (best), then summed into rfm_score (range 3-15)",
     "business_meaning": "The classic marketing RFM composite score, letting a single number stand in "
                          "for \"how good is this customer, overall\" the way an analyst would read it.",
     "churn_hypothesis": "A low rfm_score should correspond to high churn risk; this is intentionally "
                          "a coarser, more interpretable summary of information the three raw RFM "
                          "features already capture — expect it to be useful mainly for reporting and "
                          "segmentation (Step 13), not as a top SHAP feature over the raw columns.",
     "leakage_risk": "REAL RISK IF MISHANDLED: the bin edges are quantiles of the TRAINING data. "
                      "`CustomerFeatureEngineer.fit()` is called on the training split only, and the "
                      "same fitted edges are reused (not refit) for the test split — this is the "
                      "textbook train/test contamination bug this project deliberately avoids."},
    {"name": "is_high_value", "stage": "Feature engineering (Step 6)", "category": "Derived",
     "formula": "monetary_total >= the 75th percentile of monetary_total ON THE TRAINING SPLIT",
     "business_meaning": "A simple, business-readable flag for \"top-quartile spender\" — useful "
                          "directly in the retention-priority segmentation (Step 12/13) alongside the "
                          "continuous value.",
     "churn_hypothesis": "Expected to correlate with LOWER churn, mirroring the strong monotonic "
                          "monetary-vs-churn gradient found in EDA (69.1% -> 11.9% across quintiles).",
     "leakage_risk": "REAL RISK IF MISHANDLED: same as rfm_score — the 75th-percentile threshold is "
                      "computed once on the training split and applied, not recomputed, on the test "
                      "split."},
]
