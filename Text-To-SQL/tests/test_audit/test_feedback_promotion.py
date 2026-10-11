"""Feedback-loop design: POST /v1/feedback must not call llm.add_fewshot()
automatically. Promotion is a separate, explicit step —
POST /v1/admin/feedback/{id}/approve (admin-gated) or
scripts/promote_feedback.py. Both paths are exercised here against the real
app.main.app (TestClient without `with` skips the lifespan, so no live DB is
needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.routes.feedback as feedback_module
from app.audit.feedback_store import FeedbackStore
from app.auth.jwt import create_access_token
from app.auth.models import User
from app.state import state

_SALES = User(username="alice", hashed_password="unused", role="sales_us", region="us")
_ADMIN = User(username="admin", hashed_password="unused", role="admin", region="*")

client = TestClient(main_module.app)


@pytest.fixture()
def fake_feedback_store(tmp_path, monkeypatch):
    store = FeedbackStore(str(tmp_path / "feedback_queue.json"))
    monkeypatch.setattr(feedback_module, "feedback_store", store)
    return store


@pytest.fixture()
def spy_add_fewshot(monkeypatch):
    calls = []
    monkeypatch.setattr(feedback_module.llm, "add_fewshot", lambda q, sql, **kw: calls.append((q, sql)))
    return calls


def _seed_history(query_id="q1", question="How many customers?", sql="SELECT COUNT(*) FROM customers"):
    state["history"] = [
        {"query_id": query_id, "question": question, "sql": sql, "status": "ok"}
    ]


def test_feedback_does_not_auto_promote(fake_feedback_store, spy_add_fewshot):
    _seed_history()
    token = create_access_token(_SALES)
    resp = client.post(
        "/v1/feedback",
        json={"query_id": "q1", "correct": True, "note": "looks right"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert spy_add_fewshot == []  # the whole point of this design

    pending = fake_feedback_store.list(status="pending")
    assert len(pending) == 1
    assert pending[0]["query_id"] == "q1"
    assert pending[0]["correct"] is True


def test_approve_endpoint_promotes_pending_feedback(fake_feedback_store, spy_add_fewshot):
    record = fake_feedback_store.add(
        query_id="q1", question="How many customers?",
        sql="SELECT COUNT(*) FROM customers", correct=True,
    )
    token = create_access_token(_ADMIN)
    resp = client.post(
        f"/v1/admin/feedback/{record['id']}/approve",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert spy_add_fewshot == [("How many customers?", "SELECT COUNT(*) FROM customers")]
    assert fake_feedback_store.get(record["id"])["status"] == "approved"


def test_approve_endpoint_requires_admin(fake_feedback_store, spy_add_fewshot):
    record = fake_feedback_store.add(query_id="q1", question="q", sql="SELECT 1", correct=True)
    token = create_access_token(_SALES)
    resp = client.post(
        f"/v1/admin/feedback/{record['id']}/approve",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    assert spy_add_fewshot == []


def test_approve_endpoint_404_for_unknown_id(fake_feedback_store, spy_add_fewshot):
    token = create_access_token(_ADMIN)
    resp = client.post(
        "/v1/admin/feedback/not-a-real-id/approve",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_approve_twice_is_conflict_not_double_promotion(fake_feedback_store, spy_add_fewshot):
    record = fake_feedback_store.add(query_id="q1", question="q", sql="SELECT 1", correct=True)
    token = create_access_token(_ADMIN)
    first = client.post(f"/v1/admin/feedback/{record['id']}/approve",
                        headers={"Authorization": f"Bearer {token}"})
    second = client.post(f"/v1/admin/feedback/{record['id']}/approve",
                         headers={"Authorization": f"Bearer {token}"})
    assert first.status_code == 200
    assert second.status_code == 409
    assert len(spy_add_fewshot) == 1  # not promoted twice


# ---- CLI script path --------------------------------------------------------

def test_cli_script_promotes_without_auto_promotion(tmp_path, monkeypatch):
    """scripts/promote_feedback.py is the offline equivalent of the admin
    endpoint — it must also only promote on an explicit call, and must call
    add_fewshot when it does."""
    import importlib
    store_path = str(tmp_path / "feedback_queue.json")
    store = FeedbackStore(store_path)
    record = store.add(query_id="q1", question="offline question", sql="SELECT 1", correct=True)

    import scripts.promote_feedback as promote_script
    importlib.reload(promote_script)
    monkeypatch.setattr(promote_script, "feedback_store", store)
    calls = []
    monkeypatch.setattr(promote_script.llm, "add_fewshot", lambda q, sql, **kw: calls.append((q, sql)))

    promote_script._approve_one(record["id"])
    assert calls == [("offline question", "SELECT 1")]
    assert store.get(record["id"])["status"] == "approved"
