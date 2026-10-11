"""Login/refresh routes — OAuth2 password flow + refresh-token exchange."""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..auth.jwt import AuthError, create_access_token, create_refresh_token, decode_token, verify_password
from ..auth.rate_limiter import InMemoryRateLimiter
from ..auth.user_store import PostgresUserStore
from ..config import settings
from ..db.connection import pool as db_pool

router = APIRouter()

_user_store = PostgresUserStore(db_pool)

# Module-level so it persists across requests within one process (a fresh
# instance per request would defeat the point) but is easy to swap out in
# tests: monkeypatch this module's _login_rate_limiter with an instance built
# on a fake clock to test window expiry without a real sleep.
_login_rate_limiter = InMemoryRateLimiter(
    max_attempts=settings.login_rate_limit_attempts,
    window_s=settings.login_rate_limit_window_s,
)


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/v1/auth/login")
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow. Demo credentials (see app/auth/schema.sql): alice/
    raj/admin, all with password "password123" — never reuse outside this
    sandbox.

    Rate-limited per (client IP, attempted username) — see
    app/auth/rate_limiter.py — since this is the one endpoint an unthrottled
    script could brute-force in seconds, demo credentials or not.
    """
    client_ip = request.client.host if request.client else "unknown"
    rate_limit_key = f"{client_ip}:{form.username}"
    result = _login_rate_limiter.check(rate_limit_key)
    if not result.allowed:
        retry_after_s = int(result.retry_after_s) + 1
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later.",
            headers={"Retry-After": str(retry_after_s)},
        )

    user = await _user_store.get_user_by_username(form.username)
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return {
        "access_token": create_access_token(user),
        "refresh_token": create_refresh_token(user),
        "token_type": "bearer",
    }


@router.post("/v1/auth/refresh")
async def refresh(req: RefreshRequest):
    """Exchange a still-valid refresh token for a new access/refresh pair.

    Re-looks up the user rather than trusting the refresh token's embedded
    role/region/permissions verbatim, so a role change (or account removal)
    since the refresh token was issued takes effect immediately.
    """
    try:
        claims = decode_token(req.refresh_token)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    if claims.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")
    user = await _user_store.get_user_by_username(claims.get("sub", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return {
        "access_token": create_access_token(user),
        "refresh_token": create_refresh_token(user),
        "token_type": "bearer",
    }
