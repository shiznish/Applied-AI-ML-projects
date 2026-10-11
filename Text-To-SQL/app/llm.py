"""LLM integration: SQL generation, back-translation, alignment judging, and
alternative-approach generation — all with schema-validated structured output.

Providers, selected from the environment (see config.py):
  - groq      : free tier, OpenAI-compatible API (JSON mode + Pydantic validation)
  - anthropic : Claude structured outputs via messages.parse
  - bedrock   : any Bedrock model/inference profile, via forced tool-use on
                the Converse API (opt-in only: LLM_PROVIDER=bedrock)
"""
import json
import os
import re
import time

from pydantic import BaseModel, ValidationError

from .config import settings
from .observability.tracing import llm_call_span, set_llm_usage_attributes

_SECRET_RE = re.compile(r"(gsk_|sk-)[A-Za-z0-9\-_]{10,}")


def _redact(text: str) -> str:
    """Scrub API keys / bearer tokens from any string before it's surfaced or logged."""
    scrubbed = _SECRET_RE.sub("***REDACTED***", text)
    if settings.llm_api_key:
        scrubbed = scrubbed.replace(settings.llm_api_key, "***REDACTED***")
    return scrubbed


class LLMError(Exception):
    """Normalized provider error so the API layer stays provider-agnostic."""


class LLMAuthError(LLMError):
    """Missing/invalid credentials or no credits."""


class Interpretation(BaseModel):
    description: str
    example_sql: str


class FilterSpec(BaseModel):
    """A filter named by canonical semantic dimension, not a raw physical
    column — app/semantic/resolver.py.SchemaResolver derives the physical
    table/column from `dimension` and app/semantic/value_resolver.py grounds
    `value` against what's actually stored. The router is never trusted to
    name a physical column or a stored value directly."""
    dimension: str
    operator: str    # "=", "!=", ">", ">=", "<", "<=", "relative_period", ...
    value: str


class RoutedIntent(BaseModel):
    """Structured output of the semantic router (app/semantic/router.py) —
    replaces app/text2sql/schema_retriever.py's keyword-overlap table
    selection as the normal retrieval path. Fields should be canonical
    semantic-layer names (metric/dimension/domain names from layer.yaml),
    not synonym text — app/semantic/resolver.py.SchemaResolver's canonical
    lookup is the normal path; its synonym lookup is a defensive fallback
    only for the rare case a canonical ID wasn't emitted here."""
    domains: list[str]
    metrics: list[str]
    dimensions: list[str]
    filters: list[FilterSpec]
    needs_clarification: bool
    clarification_reason: str
    interpretations: list[Interpretation]  # populated when needs_clarification is true


class GeneratedSQL(BaseModel):
    needs_clarification: bool
    # Populated when needs_clarification is true — one entry per plausible reading
    interpretations: list[Interpretation]
    sql: str                    # empty string when clarification is needed
    explanation: str
    confidence: float           # model's own estimate, 0.0 - 1.0
    tables_used: list[str]
    columns_used: list[str]


class BackTranslation(BaseModel):
    question: str               # the question this SQL answers, in plain English


class AlignmentJudgment(BaseModel):
    score: float                # 0.0 (different question) - 1.0 (same question)
    reasoning: str


# --------------------------------------------------------------------------
# Provider backends
# --------------------------------------------------------------------------

_clients: dict = {}


def _anthropic_structured(system: str, user: str, model_cls, max_tokens: int,
                          low_effort: bool = False):
    import anthropic

    if "anthropic" not in _clients:
        _clients["anthropic"] = anthropic.Anthropic()
    kwargs = {}
    if low_effort:
        kwargs["output_config"] = {"effort": "low"}
    try:
        response = _clients["anthropic"].messages.parse(
            model=settings.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=model_cls,
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        set_llm_usage_attributes(
            getattr(usage, "input_tokens", None) if usage is not None else None,
            getattr(usage, "output_tokens", None) if usage is not None else None,
        )
        return response.parsed_output
    except anthropic.AuthenticationError as e:
        raise LLMAuthError(f"Anthropic auth failed: {_redact(str(e))}") from e
    except anthropic.APIStatusError as e:
        if "credit balance" in str(e).lower():
            raise LLMAuthError(f"Anthropic account has no credits: {_redact(e.message)}") from e
        raise LLMError(f"Anthropic error ({e.status_code}): {_redact(e.message)}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Anthropic API — check your network") from e
    except TypeError as e:  # SDK raises bare TypeError when no credentials resolve
        if "authentication" in str(e).lower():
            raise LLMAuthError("No Anthropic API key configured") from e
        raise


def _bedrock_structured(system: str, user: str, model_cls, max_tokens: int):
    """AWS Bedrock (e.g. a Claude inference profile). No messages.parse-style
    helper exists on Bedrock, so structured output is obtained the same way
    _openai_structured gets it — a schema handed to the model — but via
    Bedrock's native mechanism for that: a forced tool call on the Converse
    API, with the Pydantic schema as the tool's input schema."""
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    if "bedrock" not in _clients:
        _clients["bedrock"] = boto3.client(
            "bedrock-runtime",
            region_name=settings.bedrock_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
    tool_name = "emit_result"
    try:
        response = _clients["bedrock"].converse(
            modelId=settings.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            toolConfig={
                "tools": [{"toolSpec": {
                    "name": tool_name,
                    "description": "Return the structured result for this request.",
                    "inputSchema": {"json": model_cls.model_json_schema()},
                }}],
                "toolChoice": {"tool": {"name": tool_name}},
            },
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
    except NoCredentialsError as e:
        raise LLMAuthError("No AWS credentials configured for Bedrock") from e
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("UnrecognizedClientException", "AccessDeniedException", "ExpiredTokenException"):
            raise LLMAuthError(f"Bedrock auth failed ({code}): {_redact(str(e))}") from e
        raise LLMError(f"Bedrock error ({code}): {_redact(str(e))}") from e

    usage = response.get("usage", {})
    set_llm_usage_attributes(usage.get("inputTokens"), usage.get("outputTokens"))

    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            try:
                return model_cls.model_validate(block["toolUse"]["input"])
            except ValidationError as e:
                raise LLMError(f"Bedrock tool input failed schema validation: {e}") from e
    raise LLMError("Bedrock response contained no tool-use block")


def _openai_structured(system: str, user: str, model_cls, max_tokens: int):
    """OpenAI-compatible providers (Groq, Gemini-compat, OpenRouter, Ollama):
    JSON mode + schema-in-prompt, validated with Pydantic, one repair retry."""
    import openai

    if "openai" not in _clients:
        _clients["openai"] = openai.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            max_retries=5,   # free tiers rate-limit aggressively; SDK backs off on 429
        )
    schema = json.dumps(model_cls.model_json_schema())
    messages = [
        {"role": "system", "content":
            f"{system}\n\nRespond ONLY with a single JSON object that validates "
            f"against this JSON schema — no markdown fences, no prose:\n{schema}"},
        {"role": "user", "content": user},
    ]

    last_err = None
    for _ in range(2):  # one repair retry on validation failure
        try:
            resp = _clients["openai"].chat.completions.create(
                model=settings.model,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except openai.AuthenticationError as e:
            raise LLMAuthError(
                f"{settings.llm_provider} auth failed — check the API key: {_redact(str(e))}"
            ) from e
        except openai.APIStatusError as e:
            raise LLMError(f"{settings.llm_provider} error ({e.status_code}): {_redact(str(e))}") from e
        except openai.APIConnectionError as e:
            # Surface the underlying cause (DNS / SSL / timeout / blocked egress)
            # so a deploy-environment network failure is diagnosable — but redact
            # any credential from the message first (the cause can echo the header).
            cause = e.__cause__ or e
            raise LLMError(
                f"Could not reach {settings.llm_provider} at {settings.llm_base_url} "
                f"({type(cause).__name__}: {_redact(str(cause))})"
            ) from e

        usage = getattr(resp, "usage", None)
        set_llm_usage_attributes(
            getattr(usage, "prompt_tokens", None) if usage is not None else None,
            getattr(usage, "completion_tokens", None) if usage is not None else None,
        )
        text = resp.choices[0].message.content or ""
        try:
            return model_cls.model_validate_json(text)
        except ValidationError as e:
            last_err = e
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                f"That JSON failed schema validation:\n{e}\nReturn a corrected JSON object only."})
    raise LLMError(f"Model returned invalid JSON twice: {last_err}")


def _parse_structured(system: str, user: str, model_cls, max_tokens: int,
                      low_effort: bool = False):
    if settings.llm_provider == "anthropic":
        return _anthropic_structured(system, user, model_cls, max_tokens, low_effort)
    if settings.llm_provider == "bedrock":
        return _bedrock_structured(system, user, model_cls, max_tokens)
    return _openai_structured(system, user, model_cls, max_tokens)


# --------------------------------------------------------------------------
# Few-shot examples (seed + feedback-loop learned)
# --------------------------------------------------------------------------

_FEWSHOT_PATH = os.path.join(settings.log_dir, "fewshot.json")

_SEED_FEWSHOTS = [
    {
        "question": "How many customers signed up in 2025?",
        "sql": "SELECT COUNT(*) AS customers FROM customers "
               "WHERE signup_date BETWEEN '2025-01-01' AND '2025-12-31'",
    },
    {
        "question": "Top 5 products by total revenue",
        "sql": "SELECT p.name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue "
               "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
               "GROUP BY p.name ORDER BY revenue DESC LIMIT 5",
    },
    {
        # Deliberately kept alongside the undated example above, not instead
        # of it: without this one, that example's near-identical wording was
        # strong enough to anchor generation onto the same join shape even
        # when a question added a year/date condition, dropping the orders
        # join (and the order_date column it carries) entirely rather than
        # using it.
        "question": "Top 5 products by revenue in 2025",
        "sql": "SELECT p.name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue "
               "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
               "JOIN orders o ON o.order_id = oi.order_id "
               "WHERE o.order_date BETWEEN '2025-01-01' AND '2025-12-31' "
               "GROUP BY p.name ORDER BY revenue DESC LIMIT 5",
    },
    {
        "question": "What share of orders were cancelled, by channel?",
        "sql": "SELECT channel, COUNT(*) FILTER (WHERE status = 'cancelled') * 1.0 / COUNT(*) "
               "AS cancel_rate FROM orders GROUP BY channel ORDER BY cancel_rate DESC",
    },
    {
        "question": "Average payment amount for captured credit card payments",
        "sql": "SELECT AVG(amount) AS avg_amount FROM payments "
               "WHERE method = 'credit_card' AND status = 'captured'",
    },
]


def _load_fewshots() -> list[dict]:
    """Legacy unconditional fewshot list: seed examples + newest 3 learned
    ones, no relevance filtering. Still used by this module's own
    generate_sql (the raw-schema-markdown path) and by
    app/text2sql/pipeline.py's offline/no-router fallback, where there's no
    RoutedIntent to score against. The routed primary path uses
    app/text2sql/verified_queries.py's structured-overlap scoring instead —
    see that module's docstring for why an unconditional "newest 3" is the
    wrong default once a question's semantic intent is actually known."""
    examples = list(_SEED_FEWSHOTS)
    # TEXT2SQL_EVAL_MODE pins the prompt to the fixed seed examples so eval runs
    # are reproducible regardless of prior interactive 👍 feedback.
    if os.getenv("TEXT2SQL_EVAL_MODE"):
        return examples
    if os.path.exists(_FEWSHOT_PATH):
        try:
            with open(_FEWSHOT_PATH, encoding="utf-8") as f:
                examples.extend(json.load(f)[-3:])  # newest 3 learned examples
        except (json.JSONDecodeError, OSError):
            pass
    return examples


def _load_learned_examples() -> list[dict]:
    """Every learned example, unfiltered and unscored — each carries
    question/sql/domains/metrics/dimensions/ts (see add_fewshot below).
    Consumed by app/text2sql/verified_queries.py, which does the actual
    relevance scoring against a specific question's RoutedIntent; this
    function's only job is loading the raw pool to score from."""
    if os.getenv("TEXT2SQL_EVAL_MODE"):
        return []  # eval reproducibility — same reasoning as _load_fewshots above
    if not os.path.exists(_FEWSHOT_PATH):
        return []
    try:
        with open(_FEWSHOT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def add_fewshot(question: str, sql: str, domains: list[str] | None = None,
                 metrics: list[str] | None = None, dimensions: list[str] | None = None) -> None:
    """Feedback loop: results the user marked correct become future examples.
    domains/metrics/dimensions (from the RoutedIntent that produced this
    query — see app/main.py's approve_feedback) let
    app/text2sql/verified_queries.py score this example's relevance to a
    later, different question instead of just replaying the newest ones."""
    existing = []
    if os.path.exists(_FEWSHOT_PATH):
        try:
            with open(_FEWSHOT_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append({
        "question": question, "sql": sql, "ts": time.time(),
        "domains": domains or [], "metrics": metrics or [], "dimensions": dimensions or [],
    })
    with open(_FEWSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing[-50:], f, indent=2)


def _system_prompt(schema_md: str) -> str:
    shots = "\n\n".join(
        f"Q: {ex['question']}\nSQL: {ex['sql']}" for ex in _load_fewshots()
    )
    return f"""You are an expert PostgreSQL SQL analyst. Translate the user's natural
language question into a single read-only PostgreSQL SELECT query against this schema:

{schema_md}

Rules:
- SELECT queries only. Never write INSERT/UPDATE/DELETE/DDL.
- Use only tables and columns that exist in the schema above.
- Revenue for an order item = quantity * unit_price * (1 - discount).
- Prefer explicit JOINs following the foreign keys shown.
- Include a sensible LIMIT for queries that could return many rows.
- If the question is genuinely ambiguous (a term maps to multiple plausible
  interpretations, e.g. "revenue" could be gross vs net of refunds), set
  needs_clarification=true, leave sql empty, and list each interpretation with
  an example query. Only do this for real ambiguity — not for minor wording.
- If the question cannot be answered from this schema at all, set
  needs_clarification=true and explain why in the interpretations.
- Report an honest confidence (0-1) that your SQL answers the question.

Examples:
{shots}"""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def generate_sql(question: str, schema_md: str) -> GeneratedSQL:
    with llm_call_span("generate_content"):
        out = _parse_structured(_system_prompt(schema_md), question, GeneratedSQL, 2000)
    out.confidence = min(1.0, max(0.0, out.confidence))
    return out


def back_translate(sql: str, schema_md: str) -> str:
    """Hallucination check, first pass: what question does this SQL actually answer?"""
    system = ("You translate SQL queries back into the plain-English question they "
              "answer. Be precise about filters, groupings and metrics. Schema for "
              f"context:\n\n{schema_md}")
    with llm_call_span("back_translate"):
        out = _parse_structured(system, f"What question does this SQL query answer?\n\n{sql}",
                                BackTranslation, 500, low_effort=True)
    return out.question


def judge_alignment(original_question: str, back_translated: str) -> AlignmentJudgment:
    """Hallucination check, second pass: does the back-translated question match the original?"""
    system = ("You judge whether two questions ask for the same thing. Score 1.0 when "
              "they would be answered by identical data, ~0.5 when related but with "
              "different filters/metrics/groupings, 0.0 when unrelated. Wording "
              "differences don't matter; semantic differences do.")
    user = (f"Question A (what the user asked):\n{original_question}\n\n"
            f"Question B (what the generated SQL actually answers):\n{back_translated}")
    with llm_call_span("judge_alignment"):
        out = _parse_structured(system, user, AlignmentJudgment, 500, low_effort=True)
    out.score = min(1.0, max(0.0, out.score))
    return out
