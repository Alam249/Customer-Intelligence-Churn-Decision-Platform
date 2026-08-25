-- =============================================================================
-- Build the churn labels and the analytical feature table for one cutoff.
--
-- Parameters (supplied with psql -v):
--   :cutoff_date     e.g. '2011-06-09'  — end of the observation window
--   :horizon_days    e.g. 183           — label window length after the cutoff
--   :lookback_days   e.g. 365           — customer must have bought in this
--                                         window before the cutoff to be eligible
--
-- Run:
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
--        -v cutoff_date="'2011-06-09'" -v horizon_days=183 -v lookback_days=365 \
--        -f sql/build_features.sql
--
-- LEAKAGE CONTRACT
-- ----------------
-- Every feature CTE below filters `invoices.invoice_ts < :cutoff_date::date + 1`
-- (i.e. the cutoff day inclusive). The label CTE is the ONLY place that reads
-- rows after the cutoff, and it reads nothing except the existence of a sale.
-- sql/validation.sql asserts this holds after the fact.
-- =============================================================================

\set ON_ERROR_STOP on

BEGIN;

-- Cutoff-day-inclusive upper bound, used by every feature CTE.
CREATE TEMP TABLE params AS
SELECT :cutoff_date::date                                AS cutoff_date,
       (:cutoff_date::date + 1)::timestamp               AS obs_end,
       :horizon_days::int                                AS horizon_days,
       :lookback_days::int                               AS lookback_days;


-- ---------------------------------------------------------------------------
-- Base fact sets, restricted to the observation window.
-- `merchandise` deliberately excludes postage, fees, discounts, samples,
-- adjustments and test rows so that monetary features measure product revenue.
-- ---------------------------------------------------------------------------
CREATE TEMP TABLE merchandise AS
SELECT i.customer_id,
       i.invoice_no,
       i.invoice_ts,
       i.invoice_ts::date AS invoice_date,
       l.stock_code,
       l.quantity,
       l.unit_price,
       l.line_revenue
FROM invoices i
JOIN invoice_lines l USING (invoice_no)
JOIN products     p ON p.stock_code = l.stock_code
CROSS JOIN params pr
WHERE i.customer_id IS NOT NULL
  AND i.invoice_type = 'SALE'
  AND i.invoice_ts   < pr.obs_end
  AND p.item_type    = 'PRODUCT'
  AND l.quantity   > 0
  AND l.unit_price > 0;

CREATE INDEX ON merchandise (customer_id);

CREATE TEMP TABLE credits AS
SELECT i.customer_id,
       i.invoice_no,
       abs(l.line_revenue) AS credit_value
FROM invoices i
JOIN invoice_lines l USING (invoice_no)
CROSS JOIN params pr
WHERE i.customer_id IS NOT NULL
  AND i.invoice_type = 'CREDIT'
  AND i.invoice_ts   < pr.obs_end;

CREATE INDEX ON credits (customer_id);


-- ---------------------------------------------------------------------------
-- Eligibility + label.
-- Eligible  : bought merchandise in the `lookback_days` before the cutoff.
-- is_churned: made no purchase at all in the `horizon_days` after the cutoff.
-- ---------------------------------------------------------------------------
DELETE FROM churn_labels
WHERE cutoff_date  = (SELECT cutoff_date  FROM params)
  AND horizon_days = (SELECT horizon_days FROM params);

INSERT INTO churn_labels (customer_id, cutoff_date, horizon_days,
                          eligibility_lookback_days, is_churned)
SELECT e.customer_id,
       pr.cutoff_date,
       pr.horizon_days,
       pr.lookback_days,
       NOT EXISTS (
           SELECT 1
           FROM invoices f
           WHERE f.customer_id  = e.customer_id
             AND f.invoice_type = 'SALE'
             AND f.invoice_ts  >= pr.obs_end
             AND f.invoice_ts   < pr.obs_end + make_interval(days => pr.horizon_days)
       ) AS is_churned
FROM (
    SELECT DISTINCT m.customer_id
    FROM merchandise m
    CROSS JOIN params p
    WHERE m.invoice_ts >= p.obs_end - make_interval(days => p.lookback_days)
) e
CROSS JOIN params pr;


-- ---------------------------------------------------------------------------
-- Feature assembly.
-- ---------------------------------------------------------------------------
DELETE FROM customer_features WHERE cutoff_date = (SELECT cutoff_date FROM params);

INSERT INTO customer_features
WITH eligible AS (
    SELECT customer_id FROM churn_labels
    WHERE cutoff_date  = (SELECT cutoff_date  FROM params)
      AND horizon_days = (SELECT horizon_days FROM params)
),

-- Invoice-level roll-up: one row per (customer, invoice).
orders AS (
    SELECT m.customer_id,
           m.invoice_no,
           min(m.invoice_date)  AS invoice_date,
           sum(m.line_revenue)  AS order_value,
           sum(m.quantity)      AS order_items
    FROM merchandise m
    JOIN eligible USING (customer_id)
    GROUP BY m.customer_id, m.invoice_no
),

-- Recency / frequency / monetary / tenure / basket size.
rfm AS (
    SELECT o.customer_id,
           (SELECT cutoff_date FROM params) - max(o.invoice_date)  AS recency_days,
           count(*)                                                AS frequency,
           sum(o.order_value)                                      AS monetary_total,
           avg(o.order_value)                                      AS monetary_avg_order,
           (SELECT cutoff_date FROM params) - min(o.invoice_date)  AS tenure_days,
           count(DISTINCT o.invoice_date)                          AS active_days,
           sum(o.order_items)                                      AS total_items,
           avg(o.order_items)                                      AS avg_items_per_order
    FROM orders o
    GROUP BY o.customer_id
),

-- Gaps between consecutive purchase *days* (not invoices), so a customer who
-- raises three invoices on one day does not register two zero-day gaps.
purchase_days AS (
    SELECT DISTINCT customer_id, invoice_date FROM orders
),
gaps AS (
    SELECT customer_id,
           invoice_date - lag(invoice_date) OVER (
               PARTITION BY customer_id ORDER BY invoice_date
           ) AS gap_days
    FROM purchase_days
),
cadence AS (
    SELECT customer_id,
           avg(gap_days)::numeric    AS avg_interpurchase_days,
           stddev_samp(gap_days)::numeric AS std_interpurchase_days
    FROM gaps
    WHERE gap_days IS NOT NULL
    GROUP BY customer_id
),

-- Catalogue breadth and price point.
basket AS (
    SELECT m.customer_id,
           count(DISTINCT m.stock_code) AS distinct_products,
           -- Quantity-weighted so a 100-unit line counts more than a 1-unit line.
           (sum(m.line_revenue) / NULLIF(sum(m.quantity), 0))::numeric AS avg_unit_price
    FROM merchandise m
    JOIN eligible USING (customer_id)
    GROUP BY m.customer_id
),

-- Returns behaviour.
returns AS (
    SELECT c.customer_id,
           count(DISTINCT c.invoice_no) AS return_invoices,
           sum(c.credit_value)          AS return_value
    FROM credits c
    JOIN eligible USING (customer_id)
    GROUP BY c.customer_id
),

-- Short-window activity: the trend signal that separates a fading customer
-- from a steady one with the same lifetime totals.
recent AS (
    SELECT o.customer_id,
           count(*) FILTER (
               WHERE o.invoice_date > (SELECT cutoff_date FROM params) - 30
           ) AS orders_last_30d,
           count(*) FILTER (
               WHERE o.invoice_date > (SELECT cutoff_date FROM params) - 90
           ) AS orders_last_90d,
           COALESCE(sum(o.order_value) FILTER (
               WHERE o.invoice_date > (SELECT cutoff_date FROM params) - 90
           ), 0) AS spend_last_90d
    FROM orders o
    GROUP BY o.customer_id
)

SELECT r.customer_id,
       (SELECT cutoff_date FROM params) AS cutoff_date,

       r.recency_days,
       r.frequency,
       round(r.monetary_total, 2),
       round(r.monetary_avg_order, 2),

       r.tenure_days,
       r.active_days,
       round(cad.avg_interpurchase_days, 2),
       round(cad.std_interpurchase_days, 2),
       -- Guard tenure = 0 (single-day customers) against division by zero.
       round((r.frequency / (GREATEST(r.tenure_days, 1) / 30.44))::numeric, 4),

       r.total_items,
       round(r.avg_items_per_order, 2),
       b.distinct_products,
       round(b.avg_unit_price, 4),

       COALESCE(ret.return_invoices, 0),
       round(COALESCE(ret.return_value, 0), 2),
       round(COALESCE(ret.return_value, 0) / NULLIF(r.monetary_total, 0), 4),

       rec.orders_last_30d,
       rec.orders_last_90d,
       round(rec.spend_last_90d, 2),
       round(rec.spend_last_90d / NULLIF(r.monetary_total, 0), 4),

       co.country_name,
       co.country_name = 'United Kingdom'
FROM rfm r
JOIN basket    b   USING (customer_id)
JOIN recent    rec USING (customer_id)
LEFT JOIN cadence cad USING (customer_id)
LEFT JOIN returns ret USING (customer_id)
JOIN customers cu ON cu.customer_id = r.customer_id
JOIN countries co ON co.country_id  = cu.primary_country_id;

COMMIT;

ANALYZE customer_features;
ANALYZE churn_labels;

\echo ''
\echo '--- Label distribution ---'
SELECT cutoff_date, horizon_days,
       count(*)                                     AS customers,
       count(*) FILTER (WHERE is_churned)           AS churned,
       round(100.0 * avg(is_churned::int), 2)       AS churn_rate_pct
FROM churn_labels
GROUP BY cutoff_date, horizon_days
ORDER BY cutoff_date;

\echo ''
\echo '--- Feature table ---'
SELECT cutoff_date, count(*) AS rows FROM customer_features GROUP BY 1 ORDER BY 1;
