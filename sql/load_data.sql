-- =============================================================================
-- Load the relational CSVs produced by src/data/build_relational.py.
--
-- Uses \copy (client-side) rather than COPY (server-side) so the files are read
-- with the invoking user's permissions and no superuser access is required.
--
-- Run from the repository root:
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/load_data.sql
--
-- Load order follows the foreign-key dependency graph:
--   countries -> customers -> products -> invoices -> invoice_lines
-- =============================================================================

\set ON_ERROR_STOP on

BEGIN;

TRUNCATE invoice_lines, invoices, products, customers, countries RESTART IDENTITY CASCADE;

-- countries.country_id comes from the CSV, so the sequence must be advanced
-- afterwards or the next INSERT would collide.
\copy countries (country_id, country_name) FROM 'data/interim/countries.csv' WITH (FORMAT csv, HEADER true)
SELECT setval(pg_get_serial_sequence('countries', 'country_id'), COALESCE(MAX(country_id), 1)) FROM countries;

\copy customers (customer_id, primary_country_id) FROM 'data/interim/customers.csv' WITH (FORMAT csv, HEADER true)

\copy products (stock_code, description, item_type) FROM 'data/interim/products.csv' WITH (FORMAT csv, HEADER true)

\copy invoices (invoice_no, customer_id, country_id, invoice_ts, invoice_type) FROM 'data/interim/invoices.csv' WITH (FORMAT csv, HEADER true)

\copy invoice_lines (invoice_no, stock_code, quantity, unit_price) FROM 'data/interim/invoice_lines.csv' WITH (FORMAT csv, HEADER true)

COMMIT;

-- Refresh planner statistics so the feature queries get sensible plans on the
-- freshly loaded 1.07M-row fact table.
ANALYZE countries;
ANALYZE customers;
ANALYZE products;
ANALYZE invoices;
ANALYZE invoice_lines;

\echo ''
\echo '--- Row counts after load ---'
SELECT 'countries'     AS table_name, count(*) AS rows FROM countries
UNION ALL SELECT 'customers',     count(*) FROM customers
UNION ALL SELECT 'products',      count(*) FROM products
UNION ALL SELECT 'invoices',      count(*) FROM invoices
UNION ALL SELECT 'invoice_lines', count(*) FROM invoice_lines
ORDER BY 1;
