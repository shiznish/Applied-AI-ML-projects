"""GET /v1/history (per-user recent-queries list) and POST
/v1/history/clear (per-user reset of that same list). This is the UI's
convenience "recent queries" view, backed by the shared query log
(app/state.py) — not the audit trail (app/audit/audit_log.py), which stays
append-only and untouched by either endpoint here."""
from fastapi import APIRouter, Depends

from ..auth.rbac import get_current_user
from ..db.connection import SecurityContext
from ..state import rewrite_history_file, state

router = APIRouter()


@router.get("/v1/history")
async def get_history(security_context: SecurityContext = Depends(get_current_user)):
    own = [r for r in state["history"] if r.get("username") == security_context.username]
    return {"queries": own[-50:][::-1]}


@router.post("/v1/history/clear")
async def clear_history(security_context: SecurityContext = Depends(get_current_user)):
    """Removes only the caller's own history entries — queries.jsonl is
    shared across all users (each /v1/history call already filters to
    security_context.username), so clearing must do the same rather than
    truncating the file for everyone."""
    state["history"] = [
        r for r in state["history"] if r.get("username") != security_context.username
    ]
    rewrite_history_file()
    return {"ok": True}
