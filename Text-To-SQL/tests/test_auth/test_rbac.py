"""RBAC dependency tests — built against a minimal throwaway FastAPI app so no
live DB or the full app/main.py is required."""
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth.jwt import create_access_token
from app.auth.models import User
from app.auth.rbac import get_current_user, require_permission, security_context_from_claims
from app.db.connection import SecurityContext


def test_security_context_from_claims_builds_expected_context():
    claims = {
        "sub": "alice", "role": "sales_us", "region": "us",
        "permissions": ["query:execute"], "type": "access",
    }
    ctx = security_context_from_claims(claims)
    assert isinstance(ctx, SecurityContext)
    assert ctx.username == "alice"
    assert ctx.role == "sales_us"
    assert ctx.region == "us"
    assert ctx.permissions == frozenset({"query:execute"})
    assert ctx.postgres_role == "app_role_sales_us"


def test_security_context_from_claims_missing_claim_raises():
    from app.auth.jwt import AuthError

    with pytest.raises(AuthError):
        security_context_from_claims({"sub": "alice"})


def test_admin_wildcard_grants_every_permission():
    ctx = SecurityContext(username="admin", role="admin", region="*", permissions=frozenset({"admin:*"}))
    assert ctx.has_permission("query:execute")
    assert ctx.has_permission("audit:read")
    assert ctx.has_permission("anything:at-all")


# ---- end-to-end through a minimal app, exercising get_current_user + ---------
# ---- require_permission as real FastAPI dependencies -------------------------

_ALICE = User(username="alice", hashed_password="unused", role="sales_us", region="us")
_ADMIN = User(username="admin", hashed_password="unused", role="admin", region="*")

_app = FastAPI()


@_app.get("/whoami")
def whoami(ctx: SecurityContext = Depends(get_current_user)):
    return {"username": ctx.username, "role": ctx.role}


@_app.get("/admin-only")
def admin_only(ctx: SecurityContext = Depends(require_permission("audit:read"))):
    return {"ok": True}


client = TestClient(_app)


def test_get_current_user_rejects_missing_token():
    resp = client.get("/whoami")
    assert resp.status_code == 401


def test_get_current_user_rejects_invalid_token():
    resp = client.get("/whoami", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_get_current_user_accepts_valid_token():
    token = create_access_token(_ALICE)
    resp = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == {"username": "alice", "role": "sales_us"}


def test_require_permission_denies_without_permission():
    token = create_access_token(_ALICE)  # sales_us -> query:execute only
    resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_require_permission_allows_admin_wildcard():
    token = create_access_token(_ADMIN)
    resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
