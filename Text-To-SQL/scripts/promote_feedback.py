#!/usr/bin/env python
"""Feedback-loop promotion — the offline/CLI counterpart to
POST /v1/admin/feedback/{id}/approve in app/main.py.

A thumbs-up on a query result is written by POST /v1/feedback as a
status="pending" record (app/audit/feedback_store.py) and never
auto-promoted into the LLM's few-shot prompt. A human (developer/operator)
runs this script to review pending items and explicitly decide which ones
become future examples via app.llm.add_fewshot() — mirroring the admin
endpoint's behavior for anyone who prefers a terminal to calling the API.

Usage:
    python scripts/promote_feedback.py --list                 # show pending items
    python scripts/promote_feedback.py --approve <feedback_id> [<id> ...]
    python scripts/promote_feedback.py --approve-all          # promote every pending item
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import llm
from app.audit.feedback_store import default_store as feedback_store


def _list_pending() -> list[dict]:
    return feedback_store.list(status="pending")


def _approve_one(feedback_id: str) -> None:
    try:
        record = feedback_store.approve(feedback_id)
    except KeyError:
        print(f"  [skip] {feedback_id}: not found")
        return
    except ValueError as e:
        print(f"  [skip] {feedback_id}: {e}")
        return
    llm.add_fewshot(record["question"], record["sql"], domains=record.get("domains"),
                     metrics=record.get("metrics"), dimensions=record.get("dimensions"))
    print(f"  [ok]   {feedback_id}: promoted \"{record['question']}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list pending feedback records")
    group.add_argument("--approve", nargs="+", metavar="ID", help="approve and promote the given feedback id(s)")
    group.add_argument("--approve-all", action="store_true", help="approve and promote every pending record")
    args = parser.parse_args()

    if args.list:
        pending = _list_pending()
        if not pending:
            print("No pending feedback.")
            return
        for r in pending:
            print(f"{r['id']}  correct={r['correct']}  Q: {r['question']!r}")
            print(f"    SQL: {r['sql']}")
            if r.get("note"):
                print(f"    note: {r['note']}")
        return

    if args.approve:
        for fid in args.approve:
            _approve_one(fid)
        return

    if args.approve_all:
        pending = _list_pending()
        if not pending:
            print("No pending feedback.")
            return
        for r in pending:
            _approve_one(r["id"])


if __name__ == "__main__":
    main()
