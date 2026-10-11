-- =============================================================================
-- app/db/policies.sql — Postgres-level role isolation + row-level security +
-- column masking.
-- =============================================================================
-- Run AFTER app/auth/schema.sql (this file's GRANTs assume the `roles` table
-- already exists, and its role names — sales_us / sales_india / admin — mirror
-- the ones seeded there). Kept in its own file rather than merged into
-- schema.sql so each phase's work stays easy to review/revert independently.
--
-- Run with the admin/owner DSN (see config.database_admin_url), e.g.:
--   psql "$DATABASE_ADMIN_URL" -f app/auth/schema.sql
--   psql "$DATABASE_ADMIN_URL" -f app/db/policies.sql
--
-- Mechanism this depends on (see app/db/connection.py's
-- _ScopedConnection.__aenter__):
--   1. Every request-scoped connection issues
--        SELECT set_config('app.current_region', <region>, true)
--      and
--        SET LOCAL ROLE "app_role_<business_role>"
--      before any query text from the pipeline/guardrails ever runs.
--   2. From that point on, Postgres's own grant system and row-level-security
--      policies apply regardless of what SQL text was generated — the LLM
--      never has to know tenancy exists, and a guardrail bug still can't cross
--      a region boundary, because the underlying role has no ability to see
--      rows/columns outside its grant, full stop.
--
-- The app itself always connects as a single DB user (the one in
-- DATABASE_URL) which must have been granted the ability to SET ROLE into
-- each of app_role_sales_us / app_role_sales_india / app_role_admin, e.g.:
--   GRANT app_role_sales_us, app_role_sales_india, app_role_admin TO text2sql;
-- (not scripted here since the app's connecting username is deployment-
-- specific; add that GRANT once per environment after running this file.)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. One NOLOGIN Postgres role per business role.
-- -----------------------------------------------------------------------------
-- NOLOGIN because nothing ever connects to Postgres *as* these roles directly
-- — the app authenticates once as its own DB user and then SET LOCAL ROLEs
-- into one of these per request. None of them are granted INSERT/UPDATE/
-- DELETE/TRUNCATE on anything, ever: the app-layer guardrails
-- (app/guardrails/sql_validator.py) already reject non-SELECT statements
-- before SQL reaches here, but per this project's defense-in-depth pattern
-- (guardrails + read-only connection + RLS, each independent of the others),
-- these roles have no write privilege to fall back on even if a guardrail
-- bug ever let a write-shaped statement through.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_role_sales_us') THEN
        CREATE ROLE app_role_sales_us NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_role_sales_india') THEN
        CREATE ROLE app_role_sales_india NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_role_admin') THEN
        CREATE ROLE app_role_admin NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO app_role_sales_us, app_role_sales_india, app_role_admin;

-- -----------------------------------------------------------------------------
-- 2. Table-by-table access decisions.
-- -----------------------------------------------------------------------------
-- Region-scoped via row-level security (these all trace back to a specific
-- order, which is where the region actually lives):
--   orders                — has its own `region` column; the source of truth.
--   order_items           — one order's line items; joins back via order_id.
--   payments              — one order's payment; joins back via order_id.
--   shipments             — one order's shipment; joins back via order_id.
--   campaign_attribution  — links one order to a campaign; joins via order_id.
--   returns               — a return against one order's line item; joins via
--                           order_items.order_id -> orders (two-hop, same
--                           reasoning: a return is order-specific transactional
--                           data, so it gets the same isolation as the order
--                           it came from).
--
-- Column-masked (not row-restricted — every customer is visible, but the PII
-- column is not) via a differential-grant view, see section 3:
--   customers              — base table restricted to app_role_admin only;
--                            non-admin roles get customers_masked instead.
--
-- Deliberately left unrestricted (no RLS, granted to every role as-is):
--   suppliers              — vendor/catalog reference data, not customer- or
--                            region-specific; a US rep needing to know which
--                            supplier stocks a product isn't a tenancy leak.
--   warehouses              — fulfillment-center reference data; same reasoning.
--   products                — product catalog; global, not region-partitioned
--                            business data.
--   marketing_campaigns     — campaign definitions (name/channel/budget/dates);
--                            campaign_attribution (which LINKS a campaign to a
--                            specific order) is region-scoped, but the campaign
--                            definition itself isn't customer data.
--   employees               — staff directory. It has its own `region` column,
--                            but it is independent, randomly-assigned metadata
--                            (see data/seed.py) rather than a foreign key back
--                            to a specific customer/order, so it isn't part of
--                            order-visibility tenancy. Judgment call: left
--                            open for now; revisit if per-region staff privacy
--                            is ever required.
--   product_reviews          — a customer's review of a product. It references
--                            customer_id, not order_id, and reviews are
--                            plausibly public-facing content (this is the same
--                            reasoning as "product ratings aren't a tenancy
--                            boundary"), so left unrestricted. If per-region
--                            customer privacy for reviews is desired later, add
--                            an EXISTS-against-customers policy the same shape
--                            as customers_masked's grant split.
-- -----------------------------------------------------------------------------

GRANT SELECT ON
    suppliers, warehouses, products, marketing_campaigns, employees, product_reviews,
    orders, order_items, payments, shipments, campaign_attribution, returns
    TO app_role_sales_us, app_role_sales_india, app_role_admin;

-- customers base table: admin only (see section 3 for the masked view the
-- other two roles use instead).
GRANT SELECT ON customers TO app_role_admin;

-- -----------------------------------------------------------------------------
-- 3. Row-level security: region isolation.
-- -----------------------------------------------------------------------------
-- `current_setting('app.current_region', true)` reads the session variable
-- set by app/db/connection.py's _ScopedConnection before any query runs; the
-- `true` (missing_ok) argument makes it return NULL rather than raising when
-- unset, so a connection that was never scoped (security_context=None, e.g.
-- internal/startup queries in app/main.py's lifespan) sees no rows rather than
-- erroring — a safe default-deny rather than a crash. The `= '*'` branch is
-- how the seeded admin user's region ('*' — see app/auth/schema.sql) bypasses
-- the filter entirely.

ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_isolation ON orders
    USING (
        region = current_setting('app.current_region', true)
        OR current_setting('app.current_region', true) = '*'
    );

ALTER TABLE order_items ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_isolation ON order_items
    USING (
        EXISTS (
            SELECT 1 FROM orders o
            WHERE o.order_id = order_items.order_id
              AND (
                  o.region = current_setting('app.current_region', true)
                  OR current_setting('app.current_region', true) = '*'
              )
        )
    );

ALTER TABLE payments ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_isolation ON payments
    USING (
        EXISTS (
            SELECT 1 FROM orders o
            WHERE o.order_id = payments.order_id
              AND (
                  o.region = current_setting('app.current_region', true)
                  OR current_setting('app.current_region', true) = '*'
              )
        )
    );

ALTER TABLE shipments ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_isolation ON shipments
    USING (
        EXISTS (
            SELECT 1 FROM orders o
            WHERE o.order_id = shipments.order_id
              AND (
                  o.region = current_setting('app.current_region', true)
                  OR current_setting('app.current_region', true) = '*'
              )
        )
    );

ALTER TABLE campaign_attribution ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_isolation ON campaign_attribution
    USING (
        EXISTS (
            SELECT 1 FROM orders o
            WHERE o.order_id = campaign_attribution.order_id
              AND (
                  o.region = current_setting('app.current_region', true)
                  OR current_setting('app.current_region', true) = '*'
              )
        )
    );

-- returns is two hops from orders (returns.order_item_id -> order_items ->
-- orders), so its policy joins through order_items rather than duplicating
-- order_items' own EXISTS subquery inline.
ALTER TABLE returns ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_isolation ON returns
    USING (
        EXISTS (
            SELECT 1 FROM order_items oi
            JOIN orders o ON o.order_id = oi.order_id
            WHERE oi.order_item_id = returns.order_item_id
              AND (
                  o.region = current_setting('app.current_region', true)
                  OR current_setting('app.current_region', true) = '*'
              )
        )
    );

-- Note: RLS policies apply to every role by default except the table owner
-- (who bypasses RLS entirely unless the table has `FORCE ROW LEVEL SECURITY`).
-- app_role_sales_us / app_role_sales_india / app_role_admin are all non-owner
-- roles here, so the policy above governs all three uniformly; admin's
-- all-region visibility comes entirely from the `= '*'` clause matching its
-- seeded region, not from any RLS bypass.

-- -----------------------------------------------------------------------------
-- 4. Column masking: customers.email
-- -----------------------------------------------------------------------------
-- Honest tradeoff, documented rather than hidden: this is view-based masking
-- plus differential GRANTs, not Snowflake/Postgres-native dynamic data
-- masking. Postgres has no first-class "mask this column for these roles"
-- primitive; a policy predicate (RLS) governs which ROWS are visible, not what
-- a cell's value is transformed to, and expressing "true admin, false
-- redacted" inside a plain view definition against a session GUC works fine at
-- SELECT time (current_setting is evaluated per-query) but every role that can
-- see the view sees the SAME (correct) row set with the SAME expression
-- result, which is exactly what we want here since the redaction should be
-- unconditional for non-admin roles rather than session-dependent. So instead
-- of one view with a role-sensing CASE, we use the simpler and more robust
-- construction: customers_masked ALWAYS redacts email, and only admin is
-- granted SELECT on the real customers table at all. A non-admin role has no
-- path to the unmasked email under any query, including one against
-- information_schema or pg_catalog, because it was never granted SELECT on
-- the base table — the masking isn't just "the view redacts it", it's
-- "the underlying data isn't visible to that role in the first place".
CREATE OR REPLACE VIEW customers_masked AS
SELECT
    customer_id,
    name,
    '***REDACTED***'::VARCHAR AS email,
    country,
    segment,
    signup_date
FROM customers;

-- customers_masked itself is not region-scoped (customers aren't tied to a
-- single order's region the way orders/order_items/etc. are — a US-based rep
-- may legitimately need to look up any customer's name/country/segment, just
-- never their real email), so no RLS policy is added here. If per-region
-- customer visibility is desired later, add a policy on customers joining
-- back through orders the same way order_items/payments/shipments do above,
-- then re-derive customers_masked from the restricted base table.
GRANT SELECT ON customers_masked TO app_role_sales_us, app_role_sales_india;
-- app_role_admin already has SELECT on the unmasked `customers` table
-- (section 2); granting it customers_masked too is harmless but unnecessary,
-- so it's deliberately omitted — admin queries should go against `customers`.
