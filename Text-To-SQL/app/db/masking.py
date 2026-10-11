"""Column-masking table-identity resolution — the application-layer
counterpart to the customers/customers_masked view split defined in
app/db/policies.sql (section 4).

Non-admin Postgres roles are never granted SELECT on the base `customers`
table at all — only on `customers_masked`, which redacts email and is
otherwise column-identical. That grant split is enforced entirely at the
database layer and is correct as designed; but anything in the application
that decides which physical table to introspect, prompt the LLM with, or
run a lookup query against has to make the identical substitution for a
non-admin caller, or that role simply cannot query customer data at all —
any reference to the literal name "customers" fails with a Postgres
permission error before RLS or masking ever gets a chance to apply.

Kept as a plain name -> name map (today: one entry) rather than a more
generic masking-policy abstraction, since the view split itself is
table-specific and declared once in policies.sql; extend the map here only
if a second masked view is ever added there.
"""
from .connection import SecurityContext

MASKED_TABLE_VIEWS: dict[str, str] = {
    "customers": "customers_masked",
}


def resolve_table_name(table: str, security_context: "SecurityContext | None") -> str:
    """The table/view name to actually query, introspect, or prompt against
    for `table`, given the caller's role.

    `security_context=None` is treated the same as admin — it marks an
    internal/system call that already runs as the admin-equivalent context
    on purpose (mirrors app/main.py's `_SYSTEM_CONTEXT` and
    app/evaluation/evaluator.py's `EVAL_SECURITY_CONTEXT`), not an
    unauthenticated one. Every other role is transparently redirected to the
    masked view it actually holds a grant on.
    """
    if security_context is None or security_context.role == "admin":
        return table
    return MASKED_TABLE_VIEWS.get(table, table)
