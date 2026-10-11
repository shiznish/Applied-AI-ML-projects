"""RequestLoggingMiddleware tests against a minimal throwaway app, writing to
a temp file rather than the real logs/ directory."""
import json
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.jwt import create_access_token
from app.auth.models import User
from app.middleware.correlation_id import CorrelationIdMiddleware
from app.middleware.logging import RequestLoggingMiddleware


def _make_client(log_path):
    app = FastAPI()
    # Correlation id must be added first so it's outermost and sets
    # request.state before the logging middleware's dispatch runs — mirrors
    # the registration order in app/main.py.
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RequestLoggingMiddleware, log_path=str(log_path))

    @app.get("/ping")
    def ping():
        return {"pong": True}

    @app.get("/boom")
    def boom():
        raise ValueError("boom")

    return TestClient(app, raise_server_exceptions=False)


def _read_records(log_path):
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_logs_method_path_status_and_latency(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    client = _make_client(log_path)

    resp = client.get("/ping")
    assert resp.status_code == 200

    records = _read_records(log_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["method"] == "GET"
    assert rec["path"] == "/ping"
    assert rec["status_code"] == 200
    assert rec["latency_ms"] >= 0
    assert rec["correlation_id"]  # set by CorrelationIdMiddleware
    assert rec["username"] is None  # no Authorization header


def test_logs_username_best_effort_when_bearer_token_present(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    client = _make_client(log_path)
    token = create_access_token(User(username="alice", hashed_password="x", role="sales_us", region="us"))

    client.get("/ping", headers={"Authorization": f"Bearer {token}"})

    records = _read_records(log_path)
    assert records[-1]["username"] == "alice"


def test_does_not_fail_request_on_invalid_bearer_token(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    client = _make_client(log_path)

    resp = client.get("/ping", headers={"Authorization": "Bearer garbage"})

    assert resp.status_code == 200
    records = _read_records(log_path)
    assert records[-1]["username"] is None


def test_logs_5xx_status_when_handler_raises(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    client = _make_client(log_path)

    resp = client.get("/boom")

    assert resp.status_code == 500
    records = _read_records(log_path)
    assert records[-1]["status_code"] == 500
