"""Central configuration, loaded once from environment / .env."""
import json
import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_logger = logging.getLogger("text2sql.config")

# NEVER use this outside of local/dev — it's stable across process restarts on
# purpose (so dev tokens survive a reload) but is checked into no secret store
# and is trivially guessable to anyone who reads this file.
_INSECURE_DEV_JWT_SECRET = "insecure-dev-secret-DO-NOT-USE-IN-PRODUCTION-change-me"


@dataclass
class GuardrailConfig:
    allow_only_select: bool = True
    max_subquery_depth: int = 3
    default_row_limit: int = int(os.getenv("TEXT2SQL_ROW_LIMIT", "1000"))
    query_timeout_s: float = float(os.getenv("TEXT2SQL_QUERY_TIMEOUT_S", "10"))
    # --- app/guardrails/cost_guard.py thresholds -----------------------------
    # Complexity heuristics, not a real cost model — see that module's
    # docstring. Defaults are chosen for the project's ~12-15 table demo
    # schema: a legitimate multi-table analytical question (e.g. "revenue by
    # product category for UAE customers last quarter") joins maybe 3-5
    # tables, so 4 gives headroom without waving through an accidental
    # cartesian join across most of the catalog.
    max_join_count: int = int(os.getenv("TEXT2SQL_MAX_JOIN_COUNT", "4"))
    # Postgres planner cost units (not milliseconds) from EXPLAIN's top-line
    # `cost=..total`. On this project's small seeded dataset, legitimate
    # aggregate queries land in the tens-to-low-thousands; 100000 is set high
    # enough to clear those comfortably while still catching a runaway
    # unindexed cross join or missing-join-condition cartesian product.
    max_query_cost: float = float(os.getenv("TEXT2SQL_MAX_QUERY_COST", "100000"))
    # Planner-estimated result rows (EXPLAIN's `rows=N`) before the LIMIT
    # guardrail.py injects would apply at execution time — this catches a
    # cartesian-product-shaped query at the planning stage, before Postgres
    # spends any work materializing rows only to truncate them later.
    max_query_rows: int = int(os.getenv("TEXT2SQL_MAX_QUERY_ROWS", "1000000"))
    blocked_functions: frozenset = field(default_factory=lambda: frozenset({
        # superuser file I/O (data exfiltration / disclosure)
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_ls_logdir",
        "pg_ls_waldir", "lo_import", "lo_export", "lo_get", "lo_put",
        # cross-database / network access
        "dblink", "dblink_exec", "dblink_connect",
        # environment / settings / secrets disclosure
        "current_setting", "set_config", "pg_read_server_files",
        # session/role hijacking (would bypass the RLS session variables this
        # project relies on for access control — see app/db/connection.py)
        "set_session_authorization", "pg_terminate_backend", "pg_cancel_backend",
        "pg_reload_conf",
        # arbitrary code execution surfaces some installs expose
        "pg_execute_server_program",
        # timing side-channel / DoS: a conditional sleep lets response latency
        # leak boolean facts about rows outside a caller's RLS scope, and
        # chained calls are a trivial resource-exhaustion vector
        "pg_sleep", "pg_sleep_for", "pg_notify", "dblink_send_query",
    }))


def _detect_provider() -> str:
    """Explicit LLM_PROVIDER wins; otherwise pick by which key is configured."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "anthropic"


_PROVIDER_DEFAULTS = {
    # provider: (base_url, key_env, default_model)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openai_compatible": (os.getenv("LLM_BASE_URL", ""), "LLM_API_KEY", ""),
    "anthropic": ("", "ANTHROPIC_API_KEY", "claude-sonnet-5"),
    # Bedrock has no single API key env var (AWS access key + secret instead,
    # read separately below) and no safe default model — an inference profile
    # ARN is account-specific — so TEXT2SQL_MODEL is required for this provider.
    "bedrock": ("", "", ""),
}


@dataclass
class Settings:
    llm_provider: str = _detect_provider()
    llm_base_url: str = ""
    llm_api_key: str = ""
    model: str = ""
    # --- app/llm.py Bedrock provider (LLM_PROVIDER=bedrock) ---------------------
    bedrock_region: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://text2sql:text2sql@localhost:5432/text2sql"
    )
    # Superuser/owner DSN used only by data/seed.py and app/db/policies.sql to create
    # roles/policies at setup time — the app itself never connects with this.
    database_admin_url: str = os.getenv("DATABASE_ADMIN_URL", "")
    log_dir: str = os.path.join(ROOT, "logs")
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    # Below this combined confidence the UI shows a "low confidence — verify"
    # warning. This is a hand-chosen default, NOT calibrated against labelled
    # data — it has not been tuned to a target precision/recall on flagging wrong
    # answers. Treat it as a conservative starting point; override per deployment
    # via TEXT2SQL_LOW_CONF_THRESHOLD, and see the README for the calibration
    # method that would replace this magic number with a derived one.
    low_confidence_threshold: float = float(os.getenv("TEXT2SQL_LOW_CONF_THRESHOLD", "0.6"))
    # --- app/auth JWT settings --------------------------------------------------
    # HMAC secret signing access/refresh tokens. TEXT2SQL_JWT_SECRET MUST be set
    # to a long random value in any non-local deployment — the fallback below is
    # public (it's in this source file) and only exists so the app boots and is
    # testable without any setup.
    jwt_secret: str = os.getenv("TEXT2SQL_JWT_SECRET", _INSECURE_DEV_JWT_SECRET)
    jwt_algorithm: str = os.getenv("TEXT2SQL_JWT_ALGORITHM", "HS256")
    jwt_access_expiry_minutes: int = int(os.getenv("TEXT2SQL_JWT_EXPIRY_MINUTES", "30"))
    jwt_refresh_expiry_minutes: int = int(
        os.getenv("TEXT2SQL_JWT_REFRESH_EXPIRY_MINUTES", str(7 * 24 * 60))
    )
    # --- app/auth/rate_limiter.py (POST /v1/auth/login brute-force guard) ------
    # Max login attempts per (client IP, attempted username) per window before
    # a 429 is returned. Defaults chosen to be generous enough that normal
    # human retry (mistyped password once or twice) and the test suite's
    # occasional wrong-password-then-right-password sequence never trip it,
    # while still stopping an unthrottled brute-force script cold.
    login_rate_limit_attempts: int = int(os.getenv("TEXT2SQL_LOGIN_RATE_LIMIT", "5"))
    login_rate_limit_window_s: float = float(os.getenv("TEXT2SQL_LOGIN_RATE_WINDOW_S", "60"))
    # --- app/observability/tracing.py -------------------------------------------
    # OTLP/HTTP collector endpoint (full path, incl. /v1/traces) — points at the
    # Jaeger service in docker-compose.yml by default. See tracing.py's module
    # docstring for why HTTP rather than gRPC was chosen.
    otlp_endpoint: str = os.getenv("TEXT2SQL_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    # --- app/agent/graph.py (governed agentic layer) ----------------------------
    # Hard cap on merge_intent_node invocations within one session before the
    # graph gives up with status="error" instead of looping — a defensive bound
    # against a pathological clarification back-and-forth (or a stuck/ambiguous
    # LLM), independent of any per-request timeout.
    text2sql_max_agent_steps: int = int(os.getenv("TEXT2SQL_MAX_AGENT_STEPS", "6"))
    # --- app/agent/memory.py (long-term conversation memory) -------------------
    # Rough token budget for how much raw transcript a prompt-building call
    # (merge_intent/classify_message/respond_conversationally/reformulate) is
    # allowed to render before app/agent/memory.py's summarize_node folds the
    # oldest turns into a running LLM summary instead. 3000 is a conservative
    # default for this project's small system prompts (catalog + slots +
    # transcript all share the same context window) — generous enough that an
    # ordinary multi-turn session never triggers summarization, but bounded
    # enough that an hours-long session doesn't grow the prompt unboundedly.
    agent_history_token_budget: int = int(os.getenv("TEXT2SQL_HISTORY_TOKEN_BUDGET", "3000"))
    # --- app/prompts/*/  (versioned prompt registry) ---------------------------
    # Per-family override for which prompt version is served, keyed by family
    # name (e.g. "intent", "merge", "sql_generation" — matches each
    # app/prompts/<family>/ package). Empty by default: every family serves its
    # own DEFAULT_VERSION, bumped in code via a normal reviewed PR when promoting
    # a new prompt version — the git-native analogue of a prompt hub's
    # "Production" label. This override exists only for a no-redeploy pin or
    # rollback (e.g. TEXT2SQL_PROMPT_VERSION_OVERRIDES='{"merge":"v1"}' to revert
    # one misbehaving family without reverting the code that shipped v2), not as
    # the normal promotion path.
    prompt_version_overrides: dict[str, str] = field(
        default_factory=lambda: json.loads(os.getenv("TEXT2SQL_PROMPT_VERSION_OVERRIDES", "{}"))
    )


settings = Settings()

if settings.jwt_secret == _INSECURE_DEV_JWT_SECRET:
    _logger.warning(
        "TEXT2SQL_JWT_SECRET is not set — using an insecure, publicly-known "
        "development default. Tokens signed with it can be forged by anyone "
        "who has read this source file. Set TEXT2SQL_JWT_SECRET before "
        "deploying anywhere but a local sandbox."
    )

_base_url, _key_env, _default_model = _PROVIDER_DEFAULTS.get(
    settings.llm_provider, _PROVIDER_DEFAULTS["openai_compatible"]
)
settings.llm_base_url = os.getenv("LLM_BASE_URL", _base_url)
# .strip() defends against a trailing newline/space in the env var (a very common
# copy-paste artifact in hosting dashboards) — an unstripped "\n" makes httpx
# reject the Authorization header with LocalProtocolError.
settings.llm_api_key = os.getenv(_key_env, "").strip()
settings.model = os.getenv("TEXT2SQL_MODEL") or _default_model
settings.bedrock_region = os.getenv("AWS_REGION", "us-east-1")
settings.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
settings.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
settings.database_admin_url = settings.database_admin_url or settings.database_url

os.makedirs(settings.log_dir, exist_ok=True)
