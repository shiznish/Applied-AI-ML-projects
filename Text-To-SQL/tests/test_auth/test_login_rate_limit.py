"""POST /v1/auth/login rate limiting (app/auth/rate_limiter.py).

Exercises the real app.main.app the same way tests/test_audit/
test_audit_endpoint.py does (TestClient without the `with` context manager,
so lifespan/db_pool.open() never runs — no live DB needed). The user store is
swapped for an in-memory fake so login can be exercised without Postgres, and
the module-level rate limiter is swapped for a fresh instance per test (with
an injected fake clock where a test needs to control time) so tests don't
leak attempt counts into each other or depend on real wall-clock sleeps.

The login/refresh routes (and the _user_store/_login_rate_limiter singletons
this test patches) live in app.routes.auth — see app/main.py's docstring for
why route handlers aren't in app.main itself.
"""
from typing import Optional

from fastapi.testclient import TestClient

import app.main as main_module
import app.routes.auth as auth_module
from app.auth.jwt import hash_password
from app.auth.models import User
from app.auth.rate_limiter import InMemoryRateLimiter

client = TestClient(main_module.app)


class _InMemoryUserStore:
    """Minimal UserStore fake — same pattern as
    tests/test_auth/test_user_store.py's InMemoryUserStore."""

    def __init__(self, users: list[User]):
        self._by_username = {u.username: u for u in users}

    async def get_user_by_username(self, username: str) -> Optional[User]:
        return self._by_username.get(username)


def _demo_store() -> _InMemoryUserStore:
    return _InMemoryUserStore([
        User(username="alice", hashed_password=hash_password("password123"), role="sales_us", region="us"),
    ])


class _FakeClock:
    """Controllable clock for the rate limiter — advance() moves time forward
    without a real sleep, so window-expiry tests run instantly."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _login(username: str, password: str):
    return client.post("/v1/auth/login", data={"username": username, "password": password})


def test_repeated_failed_logins_beyond_threshold_return_429(monkeypatch):
    monkeypatch.setattr(auth_module, "_user_store", _demo_store())
    monkeypatch.setattr(
        auth_module, "_login_rate_limiter",
        InMemoryRateLimiter(max_attempts=3, window_s=60, time_fn=_FakeClock()),
    )

    for _ in range(3):
        resp = _login("alice", "wrong-password")
        assert resp.status_code == 401

    resp = _login("alice", "wrong-password")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_successful_login_within_threshold_still_works(monkeypatch):
    monkeypatch.setattr(auth_module, "_user_store", _demo_store())
    monkeypatch.setattr(
        auth_module, "_login_rate_limiter",
        InMemoryRateLimiter(max_attempts=5, window_s=60, time_fn=_FakeClock()),
    )

    resp = _login("alice", "wrong-password")
    assert resp.status_code == 401

    resp = _login("alice", "password123")
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body


def test_limiter_resets_after_configured_window(monkeypatch):
    monkeypatch.setattr(auth_module, "_user_store", _demo_store())
    clock = _FakeClock()
    monkeypatch.setattr(
        auth_module, "_login_rate_limiter",
        InMemoryRateLimiter(max_attempts=2, window_s=60, time_fn=clock),
    )

    assert _login("alice", "wrong-password").status_code == 401
    assert _login("alice", "wrong-password").status_code == 401
    assert _login("alice", "wrong-password").status_code == 429

    # Move the fake clock past the window instead of sleeping for real.
    clock.advance(61)

    resp = _login("alice", "password123")
    assert resp.status_code == 200
