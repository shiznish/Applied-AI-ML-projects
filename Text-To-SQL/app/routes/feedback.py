"""User feedback (POST /v1/feedback) and its human-reviewed promotion step
(POST /v1/admin/feedback/{id}/approve). Deliberately two separate endpoints,
not one — see the docstrings below for why."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import llm
from ..audit.feedback_store import default_store as feedback_store
from ..auth.rbac import get_current_user, require_permission
from ..db.connection import SecurityContext
from ..state import state

router = APIRouter()


class FeedbackRequest(BaseModel):
    query_id: str
    correct: bool
    note: str = ""


@router.post("/v1/feedback")
async def feedback(
    req: FeedbackRequest,
    security_context: SecurityContext = Depends(get_current_user),
):
    """Writes a status="pending" feedback record — it deliberately does NOT
    call llm.add_fewshot() anymore: an unreviewed 👍 silently poisoning future
    prompts is exactly what naive auto-promotion would risk. Promotion is a
    separate, explicit, human-reviewed step — see POST
    /v1/admin/feedback/{id}/approve below (or
    scripts/promote_feedback.py for an offline/CLI equivalent)."""
    match = next((r for r in state["history"] if r.get("query_id") == req.query_id), None)
    if match is None:
        raise HTTPException(404, "query_id not found in history")

    record = feedback_store.add(
        query_id=req.query_id, question=match.get("question"), sql=match.get("sql"),
        correct=req.correct, note=req.note,
        domains=match.get("domains"), metrics=match.get("metrics"), dimensions=match.get("dimensions"),
    )
    return {"ok": True, "id": record["id"], "status": record["status"]}


@router.post("/v1/admin/feedback/{feedback_id}/approve")
async def approve_feedback(
    feedback_id: str,
    security_context: SecurityContext = Depends(require_permission("admin:*")),
):
    """Human-reviewed promotion step: moves one pending feedback record into
    llm.add_fewshot(), gated to admins since it durably changes future prompt
    behavior for every user. Never called automatically — see POST
    /v1/feedback above."""
    try:
        record = feedback_store.approve(feedback_id)
    except KeyError:
        raise HTTPException(404, "feedback_id not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    llm.add_fewshot(record["question"], record["sql"], domains=record.get("domains"),
                     metrics=record.get("metrics"), dimensions=record.get("dimensions"))
    return {"ok": True, "promoted": record}
