"""FastAPI service: natural language -> guarded SQL -> validated results.

This module is the composition root only: app construction, middleware/
tracing wiring, startup schema introspection, and router registration.
Route handlers themselves live in app/routes/, grouped by resource
(auth, query, history, feedback, admin) — see that package for the actual
request-handling logic.
"""
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import MemorySaver

from . import llm
from .agent.graph import build_graph, postgres_checkpointer
from .db.connection import SecurityContext, pool as db_pool
from .middleware.correlation_id import CorrelationIdMiddleware
from .middleware.logging import RequestLoggingMiddleware
from .observability import tracing
from .routes import admin, auth, feedback, history, query, sessions
from .schema_extract import extract_schema
from .state import load_history, state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("text2sql")


# Internal/system operations (startup schema introspection, health checks) run
# as this synthetic admin-equivalent context rather than unscoped. Without it,
# once app/db/policies.sql's RLS is applied, an unscoped connection is neither
# a table owner nor a member of any app_role_* — SELECT ... FROM orders would
# silently return zero rows (RLS default-deny), breaking startup. The
# connecting DATABASE_URL user must be granted app_role_admin (see
# app/db/policies.sql's header comment) for this SET LOCAL ROLE to succeed.
_SYSTEM_CONTEXT = SecurityContext(
    username="_system", role="admin", region="*", permissions=frozenset({"admin:*"})
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_pool.open()
    async with db_pool.scoped_connection(_SYSTEM_CONTEXT) as conn:
        state["schema"] = await extract_schema(conn)
        lo, hi = await conn.fetchrow("SELECT MIN(order_date), MAX(order_date) FROM orders")
        state["date_range"] = (str(lo), str(hi))
    state["history"] = load_history()
    logger.info("Schema loaded: %d tables, data range %s", len(state["schema"]), state["date_range"])

    # The compiled agent graph is built once and reused across every request
    # (same reasoning as db_pool: a LangGraph compiled graph is meant to be
    # invoked many times, with `thread_id` — here, the session_id — the only
    # thing distinguishing one conversation from another) — see
    # app/agent/graph.py and app/routes/query.py. A Postgres-backed
    # checkpointer is tried first so a clarification pause (interrupt())
    # survives a process restart; if it's unavailable (no live Postgres, or
    # — a real constraint on native Windows dev — psycopg's async mode
    # cannot run under the default ProactorEventLoop there), this falls back
    # to an in-memory one rather than failing startup, matching this
    # project's established graceful-degradation pattern (e.g.
    # app/text2sql/pipeline.py's offline keyword-retrieval fallback when no
    # LLM key is configured). The in-memory fallback still works correctly
    # within one process's lifetime — it only loses an in-flight
    # clarification across a restart, not within one.
    async with AsyncExitStack() as agent_stack:
        try:
            agent_checkpointer = await agent_stack.enter_async_context(postgres_checkpointer())
        except Exception as e:
            logger.warning(
                "Postgres-backed agent checkpointing unavailable (%s) — falling back to "
                "in-memory (a clarification pause will not survive a process restart).", e,
            )
            agent_checkpointer = MemorySaver()
        state["agent_graph"] = build_graph(
            db_pool, state["schema"], list(llm._SEED_FEWSHOTS), checkpointer=agent_checkpointer,
        )
        yield

    await db_pool.close()
    tracing.shutdown_tracing()


app = FastAPI(title="Text2SQL", version="0.2.0", lifespan=lifespan)

# Order matters: CorrelationIdMiddleware is added first so it ends up outermost
# (Starlette wraps middleware in reverse-registration order), meaning it sets
# request.state.correlation_id before RequestLoggingMiddleware's dispatch runs
# and can read it back.
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# OTel tracer provider + FastAPI/asyncpg auto-instrumentation. Must
# run with the `app` instance already constructed (FastAPIInstrumentor wraps
# it), but is safe to call before routes are registered below — the ASGI
# middleware stack it installs is built lazily on first request regardless of
# decorator order. Never blocks/raises even with no Jaeger running — see
# app/observability/tracing.py's module docstring.
tracing.configure_tracing(app)

app.include_router(auth.router)
app.include_router(query.router)
app.include_router(sessions.router)
app.include_router(history.router)
app.include_router(feedback.router)
app.include_router(admin.router)

_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
