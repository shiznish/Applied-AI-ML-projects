"""Semantic router — one structured-output LLM call that classifies a
question against the curated business catalog (app/semantic/layer.yaml)
*before* any table is chosen or any SQL is written.

This is the normal path's first step (app/text2sql/pipeline.py), replacing
the old flat `ambiguous_terms` floor and made the default over
app/text2sql/schema_retriever.py's keyword-overlap table selection, which
this module's own docstring documents as an explicit offline/no-LLM-key
fallback only now (kept, not deleted, since it's what makes eval's
"baseline B" arm and any no-API-key environment still work at all).

The prompt text itself (and the catalog rendering it shares with
app/agent/merge.py) lives in app/prompts/router/ and app/prompts/catalog/
— this module only does the orchestration (calling the LLM, returning the
structured result).

Structured-output contract and its real limitation
----------------------------------------------------
The router's output fields (`domains`, `metrics`, `dimensions`,
`filters[].dimension`) are supposed to be canonical semantic-layer ids, not
the raw synonym text a user typed — the whole point of curating synonyms in
layer.yaml is so the router can *understand* "customer spend" while still
emitting the canonical id `revenue`. The catalog is only known at runtime
(loaded from layer.yaml, not fixed at class-definition time), so these
fields are typed as plain `list[str]` on `llm.RoutedIntent` rather than a
`Literal[...]`/enum over the exact catalog — Pydantic has no way to express
"one of these runtime-loaded strings" as a static type annotation, and
`model_json_schema()` (what every structured-output backend in app/llm.py
actually sends the provider) is built once from the class, before any
layer.yaml has been loaded for a given process.

This is a real, documented gap, not silently worked around: nothing here
enforces at the schema level that the router can only emit ids that exist.
The prompt (see app/prompts/catalog/'s catalog_prompt) lists the exact
canonical-id catalog and instructs "canonical ids only, exactly as spelled"
as a best-effort constraint on the model's behavior — but the actual
enforcement, the part that can't misfire, is downstream: `SchemaResolver.resolve()`
(app/semantic/resolver.py) either matches a returned name to exactly one
canonical metric/dimension (via canonical lookup, with synonym lookup as a
defensive fallback) or routes it to the deterministic ambiguity gate. That
resolver-side check — not this prompt — is what makes an unrecognized or
hallucinated id fail safe instead of silently producing wrong SQL. This is
intentional defense in depth: the router is a best-effort classifier, the
resolver is the deterministic backstop.
"""
from .. import llm
from ..observability.tracing import llm_call_span
from ..prompts import router as router_prompts
from .resolver import SemanticLayer


def route(
    question: str,
    semantic_layer: SemanticLayer,
    parse_structured=llm._parse_structured,
) -> llm.RoutedIntent:
    """Classify `question` against `semantic_layer`'s catalog. `parse_structured`
    defaults to app.llm._parse_structured and is injectable — same pattern as
    app/text2sql/generator.py's `parse_structured` parameter — so this is
    unit-testable with a fake, no live LLM API key required."""
    prompts = router_prompts.active()
    system = prompts.system_prompt(semantic_layer)
    with llm_call_span("semantic_routing", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, question, llm.RoutedIntent, 1000)
    return out
