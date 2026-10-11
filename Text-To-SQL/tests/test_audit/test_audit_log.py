"""Write/read round-trip and result_hash determinism for app/audit/audit_log.py.

Uses a temp directory (tmp_path) so nothing touches the real logs/ directory —
matches this project's existing pattern for isolating file-based state in
tests (see tests/test_feedback_loop.py's llm_module fixture)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audit.audit_log import AuditLog, AuditRecord, JsonlAuditStorage, result_hash


def _make_log(tmp_path):
    return AuditLog(JsonlAuditStorage(str(tmp_path / "audit.jsonl")))


def _record(**overrides) -> AuditRecord:
    base = dict(
        correlation_id="corr-1", query_id="q1", timestamp=1234.5,
        username="alice", role="sales_us", region="us",
        question="How many orders?", status="ok",
        generated_sql="SELECT COUNT(*) FROM orders",
        executed_sql="SELECT COUNT(*) FROM orders LIMIT 1000",
        repair_attempts=0, row_count=1,
        result_hash=result_hash(["n"], [[5]]),
        confidence=0.8, confidence_breakdown={"alignment": {"value": 0.9}},
    )
    base.update(overrides)
    return AuditRecord(**base)


# ---- write / read round-trip -----------------------------------------------

def test_write_then_read_round_trips(tmp_path):
    log = _make_log(tmp_path)
    log.write(_record(query_id="q1"))
    records = log.read_all(limit=10)
    assert len(records) == 1
    assert records[0]["query_id"] == "q1"
    assert records[0]["username"] == "alice"
    assert records[0]["confidence"] == 0.8


def test_read_all_is_newest_first(tmp_path):
    log = _make_log(tmp_path)
    log.write(_record(query_id="q1"))
    log.write(_record(query_id="q2"))
    log.write(_record(query_id="q3"))
    records = log.read_all(limit=10)
    assert [r["query_id"] for r in records] == ["q3", "q2", "q1"]


def test_read_all_respects_limit(tmp_path):
    log = _make_log(tmp_path)
    for i in range(5):
        log.write(_record(query_id=f"q{i}"))
    records = log.read_all(limit=2)
    assert len(records) == 2
    assert [r["query_id"] for r in records] == ["q4", "q3"]


def test_read_all_empty_when_no_file(tmp_path):
    log = _make_log(tmp_path)
    assert log.read_all(limit=10) == []


def test_write_is_append_only_across_instances(tmp_path):
    """A fresh AuditLog pointed at the same path sees prior writes — proof
    this is durable file state, not in-memory."""
    path = str(tmp_path / "audit.jsonl")
    AuditLog(JsonlAuditStorage(path)).write(_record(query_id="q1"))
    second = AuditLog(JsonlAuditStorage(path))
    second.write(_record(query_id="q2"))
    records = second.read_all(limit=10)
    assert {r["query_id"] for r in records} == {"q1", "q2"}


# ---- result_hash determinism ------------------------------------------------

def test_result_hash_deterministic_same_input():
    h1 = result_hash(["a", "b"], [[1, "x"], [2, "y"]])
    h2 = result_hash(["a", "b"], [[1, "x"], [2, "y"]])
    assert h1 == h2


def test_result_hash_differs_on_different_rows():
    h1 = result_hash(["a"], [[1]])
    h2 = result_hash(["a"], [[2]])
    assert h1 != h2


def test_result_hash_differs_on_different_columns():
    h1 = result_hash(["a"], [[1]])
    h2 = result_hash(["b"], [[1]])
    assert h1 != h2


def test_result_hash_is_hex_sha256():
    h = result_hash(["a"], [[1]])
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex


def test_result_hash_handles_non_json_native_types():
    import datetime
    import decimal
    # Must not raise, and must be deterministic despite datetime/Decimal values.
    h1 = result_hash(["d", "amt"], [[datetime.date(2025, 1, 1), decimal.Decimal("9.99")]])
    h2 = result_hash(["d", "amt"], [[datetime.date(2025, 1, 1), decimal.Decimal("9.99")]])
    assert h1 == h2
