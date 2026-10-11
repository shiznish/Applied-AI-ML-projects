from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

DEFAULT_POLICY_FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "policies"
)


class PolicyDocumentNotFoundError(Exception):
    def __init__(self, doc_id: str) -> None:
        super().__init__(f"policy document not found: {doc_id}")
        self.doc_id = doc_id


class PolicyDocument(BaseModel):
    doc_id: str
    title: str
    version: str
    body: str


class PolicyStore:
    def __init__(self, fixture_dir: Path = DEFAULT_POLICY_FIXTURE_DIR) -> None:
        self._fixture_dir = fixture_dir

    def get(self, doc_id: str) -> PolicyDocument:
        path = self._fixture_dir / f"{doc_id}.json"
        if not path.is_file():
            raise PolicyDocumentNotFoundError(doc_id)
        record: dict[str, Any] = json.loads(path.read_text())
        return PolicyDocument.model_validate(record)


_policy_store = PolicyStore()


def register(app: FastMCP) -> None:
    @app.resource(
        "procurement://policies/{doc_id}",
        name="procurement_policy",
        mime_type="application/json",
    )
    def get_policy(doc_id: str) -> dict[str, Any]:
        return _policy_store.get(doc_id).model_dump()
