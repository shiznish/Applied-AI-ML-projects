"""DB-backed RBAC dependency wiring for FastAPI routes.

`security_context_from_claims` is the hand-off point the Postgres RLS wiring
consumes: it turns decoded JWT claims into the
`app.db.connection.SecurityContext` that `ConnectionPool.scoped_connection()`
applies as `SET LOCAL ROLE` / `app.current_region`. That object threads from
here into every query execution call in app/main.py (see app/db/policies.sql
for the RLS policies themselves), so the role/region resolved in this module
actually governs what a request can see.
"""
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from ..db.connection import SecurityContext
from .jwt import AuthError, decode_token

# tokenUrl is documentation-only for FastAPI's OpenAPI/Swagger UI (it points
# the "Authorize" button at our login route) — it is not fetched by this
# dependency itself.
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


def security_context_from_claims(claims: dict) -> SecurityContext:
    """Build a SecurityContext from decoded JWT claims.

    Claim shape (see app/auth/jwt.py:_claims_for): {"sub", "role", "region",
    "permissions": [...], "type", "iat", "exp"}.
    """
    try:
        username = claims["sub"]
        role = claims["role"]
        region = claims["region"]
        permissions = frozenset(claims.get("permissions", []))
    except KeyError as e:
        raise AuthError(f"Token is missing required claim: {e}") from e
    return SecurityContext(username=username, role=role, region=region, permissions=permissions)


async def get_current_user(token: str = Depends(_oauth2_scheme)) -> SecurityContext:
    """FastAPI dependency: decode the bearer token -> SecurityContext, or 401."""
    try:
        claims = decode_token(token)
        return security_context_from_claims(claims)
    except AuthError as e:
        raise HTTPException(
            status_code=401, detail=str(e), headers={"WWW-Authenticate": "Bearer"}
        ) from e


def require_permission(perm: str):
    """Dependency factory: `Depends(require_permission("audit:read"))`.

    Depends on get_current_user (so it also enforces authentication), then
    403s if the resolved SecurityContext lacks the given permission.
    """

    def _check(security_context: SecurityContext = Depends(get_current_user)) -> SecurityContext:
        if not security_context.has_permission(perm):
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permission: {perm}",
            )
        return security_context

    return _check
