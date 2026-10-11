-- =============================================================================
-- app/auth/schema.sql — identity/RBAC tables (users/roles/permissions)
-- =============================================================================
-- Self-contained: creates only the auth tables + seed data below. Its
-- counterpart, app/db/policies.sql (Postgres per-business-role roles, ENABLE
-- ROW LEVEL SECURITY, CREATE POLICY, masked views), is expected to be run
-- ALONGSIDE this file, not merged into it — keep the two concerns in separate
-- files so each is easy to review/revert independently. Run this file first
-- (policies.sql's per-role GRANTs reference the `roles` table this file
-- creates).
--
-- Run with the admin/owner DSN (see config.database_admin_url), e.g.:
--   psql "$DATABASE_ADMIN_URL" -f app/auth/schema.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS roles (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS permissions (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS role_permissions (
    role_name       TEXT NOT NULL REFERENCES roles(name) ON DELETE CASCADE,
    permission_name TEXT NOT NULL REFERENCES permissions(name) ON DELETE CASCADE,
    PRIMARY KEY (role_name, permission_name)
);

CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL REFERENCES roles(name),
    region          TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---- seed roles / permissions, mirroring app/auth/models.py:ROLE_PERMISSIONS -
INSERT INTO roles (name, description) VALUES
    ('sales_us', 'US sales — can run text-to-SQL queries scoped to the us region'),
    ('sales_india', 'India sales — can run text-to-SQL queries scoped to the india region'),
    ('admin', 'Full access across regions and administrative capabilities')
ON CONFLICT (name) DO NOTHING;

INSERT INTO permissions (name, description) VALUES
    ('query:execute', 'Run a natural-language question or guardrail-checked SQL'),
    ('query:raw_sql', 'Reserved: submit hand-written SQL beyond standard guardrails'),
    ('audit:read', 'Read the audit trail'),
    ('admin:*', 'Wildcard — grants every permission')
ON CONFLICT (name) DO NOTHING;

INSERT INTO role_permissions (role_name, permission_name) VALUES
    ('sales_us', 'query:execute'),
    ('sales_india', 'query:execute'),
    ('admin', 'admin:*')
ON CONFLICT DO NOTHING;

-- ---- demo users -------------------------------------------------------------
-- DEMO CREDENTIALS ONLY — never reuse these outside this sandbox. All three
-- share the plaintext password "password123", but it is bcrypt-hashed below
-- (verify with app.auth.jwt.verify_password); nothing here stores it in the
-- clear. Three roles covering every RLS boundary the app enforces:
-- alice/sales_us/us, raj/sales_india/india, admin/admin/* (all regions).
INSERT INTO users (username, hashed_password, role, region) VALUES
    ('alice', '$2b$12$Ot4uAQ6VtEkPo.IAWkq1hug7HL0su9pe8XBP8Oj.2dzG4zeipbV6e', 'sales_us', 'us'),
    ('raj',   '$2b$12$PW.yR2FR26Q3saN3q1Qr7efnw//fb3iXlqv87T4N/Km0.mbzLCBiS', 'sales_india', 'india'),
    ('admin', '$2b$12$RrlLC5JywIT7ZnbLYXTbqOYO5z.MwWyOWwHOZXnnuo5rPgcDiPOE.', 'admin', '*')
ON CONFLICT (username) DO NOTHING;
