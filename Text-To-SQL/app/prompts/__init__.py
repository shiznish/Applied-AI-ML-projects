"""System-prompt builders for every LLM call this app makes, one module per
prompt family (catalog/router/merge/intent/reformulate/interpretation/
sql_generation/safety/summarize).

Why this package exists: before it, each prompt was built inline inside the
module that uses it (app/agent/merge.py, app/agent/intent.py,
app/semantic/router.py, app/text2sql/prompts.py, ...), with no single place
to read what a given LLM call is actually told and no way to tell, after the
fact, which version of a prompt produced a given audit/eval record. Each
module here exports its builder function(s) unchanged (same signatures as
before the move) plus a `PROMPT_VERSION` string constant; callers pass that
constant into `app/observability/tracing.py`'s `llm_call_span(...,
prompt_version=...)` so it ends up as a span attribute alongside the rest of
that call's trace — git history gives the diff, the span attribute gives
after-the-fact traceability of which prompt version produced which output.
"""
