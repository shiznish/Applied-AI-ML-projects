"""Identity, JWT and DB-backed RBAC.

Scope note: this package produces a fully-populated
`app.db.connection.SecurityContext` from a validated JWT and enforces
permission checks at the API layer. `app/db/policies.sql` (threaded in via
`app/main.py` + `app/text2sql/pipeline.py`) defines the Postgres roles/RLS
policies that context is checked against, so a role/region resolved here
constrains what a request can see at the database engine level, not just at
the API layer.
"""
