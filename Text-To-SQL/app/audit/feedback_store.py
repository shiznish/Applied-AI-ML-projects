"""Feedback review queue.

A thumbs-up on a query result must never auto-promote into the LLM's
few-shot prompt (app/llm.py's add_fewshot()) — that would let a single
unreviewed "looks right" click silently poison future prompts. POST
/v1/feedback only writes a status="pending" record here; promotion into
add_fewshot() is a separate, explicit, human-reviewed step — see the
POST /v1/admin/feedback/{id}/approve route in app/main.py and the
scripts/promote_feedback.py CLI, either of which calls `approve()` below.

Storage shape: unlike audit_log.py's append-only JSONL (a true immutable
trail), this queue needs one mutable field (status: pending -> approved), so
it's a small JSON *array* file, rewritten whole on each change — the same
pattern app/llm.py already uses for logs/fewshot.json. That's a deliberate,
consistent choice: two different durability needs (immutable log vs. a small
mutable queue) get two different, already-established file shapes rather than
forcing one abstraction to do both badly. Scale here is inherently small
(human-reviewed feedback items, not every query), so whole-file rewrite is
not a concern.
"""
import dataclasses
import json
import os
import time
import uuid
from dataclasses import dataclass, field

from ..config import settings


@dataclass
class FeedbackRecord:
    id: str
    ts: float
    query_id: str
    question: str | None
    sql: str | None
    correct: bool
    note: str
    status: str = "pending"          # pending | approved
    approved_ts: float | None = None
    # RoutedIntent metadata from the original query, threaded through so an
    # approved record carries enough structure for
    # app/text2sql/verified_queries.py's overlap scoring once promoted via
    # llm.add_fewshot() — see app/main.py's /v1/feedback and
    # /v1/admin/feedback/{id}/approve.
    domains: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class FeedbackStore:
    def __init__(self, path: str):
        self.path = path

    def _read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _write_all(self, records: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)

    def add(self, query_id: str, question: str | None, sql: str | None,
            correct: bool, note: str = "", domains: list[str] | None = None,
            metrics: list[str] | None = None, dimensions: list[str] | None = None) -> dict:
        record = FeedbackRecord(
            id=uuid.uuid4().hex[:12], ts=time.time(), query_id=query_id,
            question=question, sql=sql, correct=correct, note=note,
            domains=domains or [], metrics=metrics or [], dimensions=dimensions or [],
        ).to_dict()
        records = self._read()
        records.append(record)
        self._write_all(records)
        return record

    def list(self, status: str | None = None) -> list[dict]:
        records = self._read()
        if status is not None:
            records = [r for r in records if r.get("status") == status]
        return list(reversed(records))

    def get(self, feedback_id: str) -> dict | None:
        return next((r for r in self._read() if r.get("id") == feedback_id), None)

    def approve(self, feedback_id: str) -> dict:
        """Marks a pending record approved in place. Raises KeyError if the id
        doesn't exist, ValueError if it was already approved (promotion should
        happen exactly once per record; call sites decide whether a repeat
        request is an error or a no-op)."""
        records = self._read()
        for r in records:
            if r.get("id") == feedback_id:
                if r.get("status") == "approved":
                    raise ValueError(f"feedback {feedback_id} was already approved")
                r["status"] = "approved"
                r["approved_ts"] = time.time()
                self._write_all(records)
                return r
        raise KeyError(feedback_id)


def _default_path() -> str:
    return os.path.join(settings.log_dir, "feedback_queue.json")


default_store = FeedbackStore(_default_path())
