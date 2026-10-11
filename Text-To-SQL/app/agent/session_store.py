"""Persistence for `AgentState` across turns of a conversation — see
app/db/agent_schema.sql for the table and the ownership-model rationale.

Deliberately connects via `pool.scoped_connection()` with no `SecurityContext`
(unlike the query-execution path in app/db/executor.py): this table isn't part
of the RLS-governed text2sql query path, so these queries run as the pool's
own DB user rather than one of the read-only `app_role_*` business roles,
which hold no write grants at all (see app/db/policies.sql). Ownership is
instead enforced directly in SQL with `WHERE user_id = $1` on every
statement, so a session can never be read or overwritten by any user other
than the one who created it.
"""
import json

from ..db.connection import ConnectionPool
from .state import AgentState


async def load_session(pool: ConnectionPool, session_id: str, user_id: str) -> AgentState | None:
    async with pool.scoped_connection() as conn:
        row = await conn.fetchrow(
            "SELECT state_jsonb FROM agent_sessions WHERE session_id = $1 AND user_id = $2",
            session_id,
            user_id,
        )
    if row is None:
        return None
    return json.loads(row["state_jsonb"])


async def save_session(
    pool: ConnectionPool, session_id: str, user_id: str, state: AgentState
) -> None:
    async with pool.scoped_connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_sessions (session_id, user_id, state_jsonb, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (session_id) DO UPDATE
                SET state_jsonb = EXCLUDED.state_jsonb, updated_at = now()
                WHERE agent_sessions.user_id = $2
            """,
            session_id,
            user_id,
            json.dumps(state),
        )
