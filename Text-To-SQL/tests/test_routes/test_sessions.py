"""Route-level tests for GET /v1/agent/sessions/{session_id}/history
(app/routes/sessions.py) — the endpoint app/static/index.html calls on page
load to rehydrate the chat panel after a reload.

Unlike tests/test_routes/test_agent_query.py, this mounts only the sessions
router on a bare FastAPI app and overrides `get_current_user` + monkeypatches
`load_session` directly, rather than booting the full app/main.py lifespan
(real Postgres pool/schema introspection/tracing) — session-store ownership
and the 404 contract are what this route adds on top of load_session, which
tests/test_agent/test_session_store.py already covers live against a real
database; this file is a fast, fully offline unit test of the route itself.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.state import new_session
from app.auth.rbac import get_current_user
from app.db.connection import SecurityContext
from app.routes import sessions

_ALICE = SecurityContext(username="alice", role="sales_us", region="us", permissions=frozenset())


def _make_client(fake_load_session, monkeypatch, current_user=_ALICE):
    monkeypatch.setattr(sessions, "load_session", fake_load_session)
    app = FastAPI()
    app.include_router(sessions.router)
    app.dependency_overrides[get_current_user] = lambda: current_user
    return TestClient(app)


def test_history_route_404s_for_unknown_session(monkeypatch):
    async def fake_load_session(pool, session_id, user_id):
        return None

    client = _make_client(fake_load_session, monkeypatch)
    r = client.get("/v1/agent/sessions/does-not-exist/history")
    assert r.status_code == 404


def test_history_route_returns_full_persisted_state(monkeypatch):
    state = new_session("s1")
    state["history"] = [
        {"role": "user", "content": "show revenue by country"},
        {"role": "assistant", "content": "Returned 5 row(s)."},
    ]
    state["domain"] = "sales_performance"
    state["metrics"] = ["revenue"]

    captured = {}

    async def fake_load_session(pool, session_id, user_id):
        captured["session_id"] = session_id
        captured["user_id"] = user_id
        return state

    client = _make_client(fake_load_session, monkeypatch)
    r = client.get("/v1/agent/sessions/s1/history")

    assert r.status_code == 200
    body = r.json()
    assert body["history"] == state["history"]
    assert body["domain"] == "sales_performance"
    assert body["metrics"] == ["revenue"]
    # The route must scope the lookup to the AUTHENTICATED caller, never a
    # client-supplied identity — this is what makes load_session's own
    # ownership check (WHERE user_id = $1) actually mean something here.
    assert captured["user_id"] == "alice"
    assert captured["session_id"] == "s1"


def test_history_route_is_isolated_by_user(monkeypatch):
    """A session that belongs to someone else must 404, not leak — this
    mirrors app/agent/session_store.py's own fail-safe-not-fail-open
    contract (load_session returns None for a non-owner, never raises or
    returns another user's data)."""
    async def fake_load_session(pool, session_id, user_id):
        return None if user_id != "alice" else new_session(session_id)

    bob = SecurityContext(username="bob", role="sales_india", region="india", permissions=frozenset())
    client = _make_client(fake_load_session, monkeypatch, current_user=bob)
    r = client.get("/v1/agent/sessions/s1/history")
    assert r.status_code == 404
