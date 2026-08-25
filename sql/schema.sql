-- =============================================================================
-- Customer Intelligence & Churn Decision Platform — relational schema
-- Source: UCI Online Retail II (one row per invoice line item)
-- Target: PostgreSQL 16
--
-- Design notes (each is backed by a check against the raw data):
--   * An invoice maps to exactly one customer (0 violations) and one country
--     (0 violations), so those are invoice-level attributes, not line-level.
--   * 13 of 5,942 customers transact from more than one country, so country
--     lives on `invoices`; `customers.primary_country_id` holds the modal one.
--   * (invoice_no, stock_code) is NOT unique (45,947 duplicate pairs — the same
--     item legitimately appears on multiple lines), so `invoice_lines` needs a
--     surrogate key.
--   * 1,232 stock codes carry more than one description (up to 9 variants), so
--     `products.description` stores the modal description, not a key component.
--   * Returns are modelled as invoices with invoice_type = 'CREDIT' rather than
--     a separate table — a `returns` table would duplicate the same columns.
--   * There is no subscription, payment-method, support, or marketing data in
--     this source, so no such tables are created.
-- =============================================================================

DROP TABLE IF EXISTS customer_features CASCADE;
DROP TABLE IF EXISTS churn_labels CASCADE;
DROP TABLE IF EXISTS invoice_lines CASCADE;
DROP TABLE IF EXISTS invoices CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS countries CASCADE;


-- ---------------------------------------------------------------------------
-- countries — lookup table, normalised out of the raw free-text Country column
-- ---------------------------------------------------------------------------
CREATE TABLE countries (
    country_id    SMALLSERIAL PRIMARY KEY,
    country_name  TEXT NOT NULL UNIQUE
                  CHECK (length(trim(country_name)) > 0)
);

COMMENT ON TABLE countries IS
    'Distinct shipping countries observed in the source data (43 values).';


-- ---------------------------------------------------------------------------
-- customers — one row per identified customer
-- The raw Customer ID is a stable integer (12346-18287) and is used directly
-- as the natural primary key; inventing a surrogate key would add no value.
-- ---------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id         INTEGER PRIMARY KEY
                        CHECK (customer_id > 0),
    primary_country_id  SMALLINT NOT NULL
                        REFERENCES countries (country_id) ON DELETE RESTRICT
);

COMMENT ON COLUMN customers.primary_country_id IS
    'Modal country across the customer''s invoices. 13 customers transact from '
    'more than one country; the per-invoice country is authoritative.';


-- ---------------------------------------------------------------------------
-- products — one row per stock code
-- item_type separates genuine merchandise from postage, fees and adjustments,
-- so revenue features can exclude non-merchandise without deleting the rows.
-- ---------------------------------------------------------------------------
CREATE TABLE products (
    stock_code   TEXT PRIMARY KEY
                 CHECK (stock_code = upper(trim(stock_code))),
    description  TEXT,
    item_type    TEXT NOT NULL DEFAULT 'PRODUCT'
                 CHECK (item_type IN (
                     'PRODUCT',     -- genuine merchandise
                     'POSTAGE',     -- POST, DOT, C2, C3
                     'FEE',         -- AMAZONFEE, BANK CHARGES, CRUK
                     'DISCOUNT',    -- D
                     'SAMPLE',      -- S
                     'MANUAL',      -- M
                     'VOUCHER',     -- GIFT_0001_*
                     'ADJUSTMENT',  -- ADJUST, ADJUST2, B
                     'TEST'         -- TEST001, TEST002
                 ))
);


-- ---------------------------------------------------------------------------
-- invoices — one row per invoice document
-- customer_id is nullable: 22.77% of raw rows are anonymous cash sales with no
-- customer. They are retained because they are real revenue, but they are
-- excluded from every customer-level feature by an INNER JOIN.
-- ---------------------------------------------------------------------------
CREATE TABLE invoices (
    invoice_no    TEXT PRIMARY KEY
                  CHECK (length(invoice_no) BETWEEN 6 AND 7),
    customer_id   INTEGER
                  REFERENCES customers (customer_id) ON DELETE RESTRICT,
    country_id    SMALLINT NOT NULL
                  REFERENCES countries (country_id) ON DELETE RESTRICT,
    invoice_ts    TIMESTAMP NOT NULL
                  CHECK (invoice_ts >= TIMESTAMP '2009-01-01'
                     AND invoice_ts <  TIMESTAMP '2012-01-01'),
    invoice_type  TEXT NOT NULL
                  CHECK (invoice_type IN ('SALE', 'CREDIT', 'ADJUSTMENT'))
);

COMMENT ON COLUMN invoices.invoice_ts IS
    'MIN(InvoiceDate) over the invoice''s lines. 83 invoices span more than one '
    'timestamp (median spread 60s, max 9min) — data-entry lag, not distinct events.';

COMMENT ON COLUMN invoices.invoice_type IS
    'SALE = numeric invoice number; CREDIT = "C" prefix (return/refund, 19,494 '
    'rows); ADJUSTMENT = "A" prefix (6 bad-debt write-offs, no customer).';


-- ---------------------------------------------------------------------------
-- invoice_lines — one row per line item; the fact table
-- Surrogate PK because (invoice_no, stock_code) is not unique.
-- Sign convention is preserved from the source: CREDIT lines carry negative
-- quantity. line_revenue is generated so it can never drift from its inputs.
-- ---------------------------------------------------------------------------
CREATE TABLE invoice_lines (
    line_id       BIGSERIAL PRIMARY KEY,
    invoice_no    TEXT NOT NULL
                  REFERENCES invoices (invoice_no) ON DELETE CASCADE,
    stock_code    TEXT NOT NULL
                  REFERENCES products (stock_code) ON DELETE RESTRICT,
    quantity      INTEGER NOT NULL
                  CHECK (quantity <> 0),
    unit_price    NUMERIC(12, 2) NOT NULL,
    line_revenue  NUMERIC(16, 2)
                  GENERATED ALWAYS AS (quantity * unit_price) STORED
);

COMMENT ON COLUMN invoice_lines.unit_price IS
    'Not constrained to be positive: adjustment and bad-debt lines carry '
    'negative prices. Non-positive prices are filtered in the feature layer.';


-- ---------------------------------------------------------------------------
-- churn_labels — the derived target
-- Online Retail II has no churn column. A label is (customer, cutoff, horizon):
-- a customer who purchased within `eligibility_lookback_days` before the cutoff
-- and made zero purchases in the `horizon_days` after it is churned.
-- The composite PK lets several label definitions coexist for sensitivity
-- analysis without overwriting each other.
-- ---------------------------------------------------------------------------
CREATE TABLE churn_labels (
    customer_id                INTEGER NOT NULL
                               REFERENCES customers (customer_id) ON DELETE CASCADE,
    cutoff_date                DATE NOT NULL,
    horizon_days               SMALLINT NOT NULL CHECK (horizon_days > 0),
    eligibility_lookback_days  SMALLINT NOT NULL CHECK (eligibility_lookback_days > 0),
    is_churned                 BOOLEAN NOT NULL,
    PRIMARY KEY (customer_id, cutoff_date, horizon_days)
);


-- ---------------------------------------------------------------------------
-- customer_features — the analytical feature table
-- Populated by sql/features.sql. Every column is computed strictly from
-- invoices with invoice_ts <= cutoff_date; that restriction is what prevents
-- target leakage, and it is asserted by sql/validation.sql.
-- ---------------------------------------------------------------------------
CREATE TABLE customer_features (
    customer_id              INTEGER NOT NULL,
    cutoff_date              DATE NOT NULL,

    -- RFM core
    recency_days             INTEGER NOT NULL,
    frequency                INTEGER NOT NULL,
    monetary_total           NUMERIC(16, 2) NOT NULL,
    monetary_avg_order       NUMERIC(14, 2) NOT NULL,

    -- Tenure & cadence
    tenure_days              INTEGER NOT NULL,
    active_days              INTEGER NOT NULL,
    avg_interpurchase_days   NUMERIC(10, 2),
    std_interpurchase_days   NUMERIC(10, 2),
    purchase_rate_per_month  NUMERIC(10, 4),

    -- Basket composition
    total_items              INTEGER NOT NULL,
    avg_items_per_order      NUMERIC(12, 2) NOT NULL,
    distinct_products        INTEGER NOT NULL,
    avg_unit_price           NUMERIC(12, 4),

    -- Returns behaviour
    return_invoices          INTEGER NOT NULL,
    return_value             NUMERIC(16, 2) NOT NULL,
    return_rate              NUMERIC(8, 4) NOT NULL,

    -- Recent-window activity (trend signal)
    orders_last_30d          INTEGER NOT NULL,
    orders_last_90d          INTEGER NOT NULL,
    spend_last_90d           NUMERIC(16, 2) NOT NULL,
    spend_ratio_90d          NUMERIC(10, 4),

    -- Context
    country_name             TEXT NOT NULL,
    is_uk                    BOOLEAN NOT NULL,

    PRIMARY KEY (customer_id, cutoff_date)
);


-- ---------------------------------------------------------------------------
-- Indexes
-- Chosen for the access patterns the feature queries actually use, not
-- speculatively: every feature CTE filters invoices by customer and timestamp.
-- ---------------------------------------------------------------------------
CREATE INDEX idx_invoices_customer_ts  ON invoices (customer_id, invoice_ts);
CREATE INDEX idx_invoices_ts           ON invoices (invoice_ts);
CREATE INDEX idx_invoices_type         ON invoices (invoice_type);
CREATE INDEX idx_lines_invoice         ON invoice_lines (invoice_no);
CREATE INDEX idx_lines_stock_code      ON invoice_lines (stock_code);
CREATE INDEX idx_products_item_type    ON products (item_type);
CREATE INDEX idx_churn_labels_cutoff   ON churn_labels (cutoff_date, horizon_days);
