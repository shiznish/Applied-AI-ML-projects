"""CorrelationIdMiddleware tests against a minimal throwaway app."""
import uuid

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.middleware.correlation_id import CORRELATION_ID_HEADER, CorrelationIdMiddleware

_app = FastAPI()
_app.add_middleware(CorrelationIdMiddleware)


@_app.get("/echo")
def echo(request: Request):
    return {"correlation_id": request.state.correlation_id}


client = TestClient(_app)


def test_generates_a_correlation_id_when_absent():
    resp = client.get("/echo")
    assert resp.status_code == 200
    header_id = resp.headers[CORRELATION_ID_HEADER]
    assert resp.json()["correlation_id"] == header_id
    # It should look like a uuid4 hex string.
    uuid.UUID(header_id)


def test_reuses_incoming_correlation_id_header():
    resp = client.get("/echo", headers={CORRELATION_ID_HEADER: "my-fixed-id"})
    assert resp.headers[CORRELATION_ID_HEADER] == "my-fixed-id"
    assert resp.json()["correlation_id"] == "my-fixed-id"


def test_two_requests_without_header_get_different_ids():
    id1 = client.get("/echo").headers[CORRELATION_ID_HEADER]
    id2 = client.get("/echo").headers[CORRELATION_ID_HEADER]
    assert id1 != id2
