"""Shared, process-lifetime application state: the schema/date-range/history
dict populated once at startup (app/main.py's lifespan) plus the query
history log's read/write helpers.

A plain module-level dict rather than a class or per-request dependency,
since every route module just needs read/append access to the same object
that startup populates once — passing it through FastAPI's DI for something
this simple and process-global by design would be overkill.
"""
import json
import os

from .config import settings

_QUERY_LOG = os.path.join(settings.log_dir, "queries.jsonl")

state: dict = {}


def load_history() -> list[dict]:
    if not os.path.exists(_QUERY_LOG):
        return []
    records = []
    with open(_QUERY_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-200:]


def log_query(record: dict) -> None:
    state["history"].append(record)
    with open(_QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def rewrite_history_file() -> None:
    """Rewrites queries.jsonl from state["history"] verbatim. Used by
    POST /v1/history/clear (app/routes/history.py) after filtering one
    user's entries out of state["history"] in place — the log has no
    per-line delete, so a full rewrite is how that filtering is persisted."""
    with open(_QUERY_LOG, "w", encoding="utf-8") as f:
        for record in state["history"]:
            f.write(json.dumps(record, default=str) + "\n")
