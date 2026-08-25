-- =============================================================================
-- Pipeline validation. Every row returned reports PASS or FAIL with the
-- offending count, so the output is a readable report rather than a silent
-- exception. Run after sql/build_features.sql.
--
--   psql "$DATABASE_URL" -v cutoff_date="'2011-06-09'" -f sql/validation.sql
--
-- These are *structural* checks on the pipeline. Statistical data-quality
-- profiling (distributions, outliers, class balance) is Step 4's job.
-- =============================================================================

\set ON_ERROR_STOP on

\echo ''
\echo '========== PIPELINE VALIDATION =========='
\echo ''

WITH checks AS (

    -- --- Referential integrity -------------------------------------------
    SELECT 1 AS id, 'Every invoice line references a real invoice' AS check_name,
           count(*) AS violations
    FROM invoice_lines l LEFT JOIN invoices i USING (invoice_no)
    WHERE i.invoice_no IS NULL

    UNION ALL
    SELECT 2, 'Every invoice line references a real product',
           count(*)
    FROM invoice_lines l LEFT JOIN products p USING (stock_code)
    WHERE p.stock_code IS NULL

    UNION ALL
    SELECT 3, 'Every identified invoice references a real customer',
           count(*)
    FROM invoices i LEFT JOIN customers c USING (customer_id)
    WHERE i.customer_id IS NOT NULL AND c.customer_id IS NULL

    -- --- Row-count reconciliation against the source ----------------------
    UNION ALL
    SELECT 4, 'invoice_lines row count matches source (1,067,371)',
           abs(count(*) - 1067371)
    FROM invoice_lines

    UNION ALL
    SELECT 5, 'invoices row count matches source (53,628)',
           abs(count(*) - 53628)
    FROM invoices

    UNION ALL
    SELECT 6, 'customers row count matches source (5,942)',
           abs(count(*) - 5942)
    FROM customers

    -- --- Key uniqueness ---------------------------------------------------
    UNION ALL
    SELECT 7, 'One customer per invoice (no invoice split across customers)',
           count(*)
    FROM (SELECT invoice_no FROM invoices GROUP BY invoice_no HAVING count(DISTINCT customer_id) > 1) x

    UNION ALL
    SELECT 8, 'No duplicate customers in the feature table',
           count(*)
    FROM (SELECT customer_id, cutoff_date FROM customer_features
          GROUP BY 1, 2 HAVING count(*) > 1) x

    -- --- Sign conventions -------------------------------------------------
    UNION ALL
    SELECT 9, 'CREDIT invoices carry non-positive quantities',
           count(*)
    FROM invoices i JOIN invoice_lines l USING (invoice_no)
    WHERE i.invoice_type = 'CREDIT' AND l.quantity > 0

    UNION ALL
    SELECT 10, 'SALE invoices with a known customer carry positive quantities',
           count(*)
    FROM invoices i JOIN invoice_lines l USING (invoice_no)
    WHERE i.invoice_type = 'SALE' AND i.customer_id IS NOT NULL AND l.quantity < 0

    -- --- LEAKAGE: the checks that matter most -----------------------------
    -- If any feature had been computed from post-cutoff data, the derived
    -- quantities below would be internally inconsistent with the cutoff.
    UNION ALL
    SELECT 11, 'LEAKAGE: no feature row has negative recency (purchase after cutoff)',
           count(*)
    FROM customer_features
    WHERE recency_days < 0

    UNION ALL
    SELECT 12, 'LEAKAGE: no feature row has recency > tenure',
           count(*)
    FROM customer_features
    WHERE recency_days > tenure_days

    UNION ALL
    SELECT 13, 'LEAKAGE: every eligible customer purchased within the lookback window',
           count(*)
    FROM customer_features f
    JOIN churn_labels lb USING (customer_id, cutoff_date)
    WHERE f.recency_days > lb.eligibility_lookback_days

    UNION ALL
    SELECT 14, 'LEAKAGE: labels and features cover exactly the same customers',
           count(*)
    FROM (
        SELECT customer_id FROM churn_labels
          WHERE cutoff_date = :cutoff_date::date
        EXCEPT SELECT customer_id FROM customer_features
          WHERE cutoff_date = :cutoff_date::date
        UNION ALL
        SELECT customer_id FROM customer_features
          WHERE cutoff_date = :cutoff_date::date
        EXCEPT SELECT customer_id FROM churn_labels
          WHERE cutoff_date = :cutoff_date::date
    ) x

    -- --- Feature-value sanity --------------------------------------------
    UNION ALL
    SELECT 15, 'No non-positive monetary totals in the feature table',
           count(*)
    FROM customer_features WHERE monetary_total <= 0

    UNION ALL
    SELECT 16, 'Frequency is at least 1 for every feature row',
           count(*)
    FROM customer_features WHERE frequency < 1

    UNION ALL
    SELECT 17, 'orders_last_30d never exceeds orders_last_90d',
           count(*)
    FROM customer_features WHERE orders_last_30d > orders_last_90d

    UNION ALL
    SELECT 18, 'orders_last_90d never exceeds total frequency',
           count(*)
    FROM customer_features WHERE orders_last_90d > frequency

    UNION ALL
    SELECT 19, 'active_days never exceeds frequency',
           count(*)
    FROM customer_features WHERE active_days > frequency

    UNION ALL
    SELECT 20, 'Target is present and both classes are populated',
           CASE WHEN count(DISTINCT is_churned) = 2 THEN 0 ELSE 1 END
    FROM churn_labels WHERE cutoff_date = :cutoff_date::date
),

-- Documented source anomalies. A tolerance is only ever added after the rows
-- have been inspected and shown to be a defect in the source data rather than
-- in this pipeline; the note records that finding.
tolerances (id, tolerance, note) AS (
    VALUES
    (9, 1, 'C496350: one "Manual" adjustment line booked positive on a credit '
           'note, no customer attached. Source defect; already excluded from '
           'features by the anonymous-invoice and item_type filters.')
)
SELECT c.id,
       CASE WHEN c.violations = 0                             THEN 'PASS'
            WHEN c.violations <= COALESCE(t.tolerance, 0)     THEN 'KNOWN'
            ELSE                                                   'FAIL'
       END AS status,
       c.check_name,
       c.violations,
       t.note
FROM checks c
LEFT JOIN tolerances t USING (id)
ORDER BY (c.violations > COALESCE(t.tolerance, 0)) DESC,
         (c.violations > 0) DESC,
         c.id;

\echo ''
\echo '--- Reconciliation: rows excluded from the feature layer, and why ---'
SELECT 'raw invoice lines'                     AS stage, count(*) AS rows FROM invoice_lines
UNION ALL SELECT 'lines on anonymous invoices', count(*)
    FROM invoices i JOIN invoice_lines l USING (invoice_no) WHERE i.customer_id IS NULL
UNION ALL SELECT 'lines on CREDIT invoices',    count(*)
    FROM invoices i JOIN invoice_lines l USING (invoice_no) WHERE i.invoice_type = 'CREDIT'
UNION ALL SELECT 'lines on ADJUSTMENT invoices', count(*)
    FROM invoices i JOIN invoice_lines l USING (invoice_no) WHERE i.invoice_type = 'ADJUSTMENT'
UNION ALL SELECT 'lines on non-merchandise codes', count(*)
    FROM invoice_lines l JOIN products p USING (stock_code) WHERE p.item_type <> 'PRODUCT'
UNION ALL SELECT 'lines with quantity <= 0',     count(*) FROM invoice_lines WHERE quantity <= 0
UNION ALL SELECT 'lines with unit_price <= 0',   count(*) FROM invoice_lines WHERE unit_price <= 0;
