"""GET /v1/agent/sessions/{session_id}/history — lets the frontend rehydrate
a conversation's chat panel after a page reload. app/static/index.html keeps
the agent `session_id` in localStorage so the BACKEND conversation survives a
reload (see app/agent/session_store.py, app/routes/query.py), but until this
route existed nothing re-fetched the transcript, so the chat panel itself
came back empty even though the agent still remembered everything.

Returns the full persisted `AgentState` (see app/agent/state.py) for a
session this user owns — same ownership check app/agent/session_store.py's
load_session already enforces via `WHERE user_id = $1`, so a caller can
never read another user's session by guessing its id.
"""
from fastapi import APIRouter, Depends, HTTPException

from ..agent.session_store import load_session
from ..auth.rbac import get_current_user
from ..db.connection import SecurityContext, pool as db_pool

router = APIRouter()


@router.get("/v1/agent/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    security_context: SecurityContext = Depends(get_current_user),
):
    state = await load_session(db_pool, session_id, security_context.username)
    if state is None:
        raise HTTPException(404, "No session found for this id")
    return state
