"""Static auth data model: the `User` dataclass and the role -> permissions map.

This module holds no live DB queries (see app/auth/user_store.py for that) — it
is the pure, importable-anywhere shape shared by jwt.py, rbac.py and every
UserStore implementation.
"""
from dataclasses import dataclass

# Permissions used across this codebase:
#   query:execute  - run a question/SQL through /v1/query or /v1/execute
#   query:raw_sql  - (reserved) submit hand-written SQL beyond guardrail-checked
#                    power-user use; not yet gated anywhere
#   audit:read     - read the audit trail
#   admin:*        - wildcard, grants every permission (see
#                    SecurityContext.has_permission in app/db/connection.py)
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "sales_us": frozenset({"query:execute"}),
    "sales_india": frozenset({"query:execute"}),
    "admin": frozenset({"admin:*"}),
}


@dataclass(frozen=True)
class User:
    """A resolved user record, as returned by a UserStore.

    `role` and `region` map directly onto `app.db.connection.SecurityContext` —
    `hashed_password` never leaves this dataclass once login succeeds (it is
    never embedded in a JWT or returned in an API response).
    """

    username: str
    hashed_password: str
    role: str
    region: str

    @property
    def permissions(self) -> frozenset[str]:
        return ROLE_PERMISSIONS.get(self.role, frozenset())
