-- =============================================================================
-- app/db/agent_schema.sql — session storage for the governed agentic layer
-- =============================================================================
-- Additive and separable from app/auth/schema.sql and app/db/policies.sql —
-- same reasoning as policies.sql being its own file: easy to review/revert
-- independently of the auth/RLS setup it doesn't touch. Can be run any time
-- after app/auth/schema.sql (no FK back to it; user_id is stored as the
-- username string, matching app/auth's TEXT primary key convention, but
-- deliberately not a REFERENCES — an agent session for a user who was later
-- deleted should not cascade-fail or cascade-delete; see
-- app/agent/session_store.py for how ownership is actually enforced).
--
-- Run with the admin/owner DSN, e.g.:
--   psql "$DATABASE_ADMIN_URL" -f app/db/agent_schema.sql
--
-- Ownership here is enforced in application code (every query is scoped with
-- `WHERE user_id = $1`), not via Postgres row-level security like
-- orders/order_items/etc. in policies.sql — this table isn't part of the
-- text2sql query-execution path those RLS policies and app_role_* grants
-- exist for (which must stay read-only, see policies.sql section 1), so it is
-- deliberately NOT granted to any app_role_* role. It's read/written by the
-- app's own connecting DB user only, via a connection that is never put into
-- one of those roles (app/agent/session_store.py passes no SecurityContext to
-- `pool.scoped_connection`, so no `SET LOCAL ROLE` happens for these queries).
-- =============================================================================

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    state_jsonb JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_sessions_user_id_idx ON agent_sessions (user_id);
