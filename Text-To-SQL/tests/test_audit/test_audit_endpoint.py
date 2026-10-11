"""GET /v1/admin/audit gating: 401 with no token, 403 for a non-admin role,
200 for admin — same pattern as tests/test_auth/test_rbac.py's end-to-end
dependency tests. Exercises the real app.main.app (no live DB or LLM key
needed: FastAPI's TestClient does not run the lifespan startup unless used as
a `with` context manager, so db_pool.open() never runs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.audit import audit_log
from app.audit.audit_log import AuditLog, AuditRecord, JsonlAuditStorage
from app.auth.jwt import create_access_token
from app.auth.models import User

import app.main as main_module

_SALES = User(username="alice", hashed_password="unused", role="sales_us", region="us")
_ADMIN = User(username="admin", hashed_password="unused", role="admin", region="*")

client = TestClient(main_module.app)


def test_audit_endpoint_rejects_missing_token():
    resp = client.get("/v1/admin/audit")
    assert resp.status_code == 401


def test_audit_endpoint_rejects_non_admin_role():
    token = create_access_token(_SALES)
    resp = client.get("/v1/admin/audit", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_audit_endpoint_allows_admin(tmp_path, monkeypatch):
    # Point the module's default log at a temp file so this test doesn't
    # depend on (or pollute) the real logs/audit.jsonl.
    fake_log = AuditLog(JsonlAuditStorage(str(tmp_path / "audit.jsonl")))
    fake_log.write(AuditRecord(
        correlation_id="c1", query_id="q1", timestamp=1.0,
        username="alice", role="sales_us", region="us", question="hi", status="ok",
    ))
    monkeypatch.setattr(audit_log, "default_log", fake_log)

    token = create_access_token(_ADMIN)
    resp = client.get("/v1/admin/audit", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert "records" in body
    assert any(r["query_id"] == "q1" for r in body["records"])


def test_audit_endpoint_respects_limit(tmp_path, monkeypatch):
    fake_log = AuditLog(JsonlAuditStorage(str(tmp_path / "audit2.jsonl")))
    for i in range(5):
        fake_log.write(AuditRecord(
            correlation_id="c", query_id=f"q{i}", timestamp=float(i),
            username="admin", role="admin", region="*", question="q", status="ok",
        ))
    monkeypatch.setattr(audit_log, "default_log", fake_log)

    token = create_access_token(_ADMIN)
    resp = client.get("/v1/admin/audit?limit=2", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert len(resp.json()["records"]) == 2
