"""Multi-turn intent merging — the one new LLM responsibility the agentic
layer adds on top of the existing pipeline. app/semantic/router.py classifies
a single, self-contained question against the catalog; this module does the
same classification but for a *follow-up* message that may only partially
specify intent, carrying forward anything a prior turn already resolved that
the new message doesn't contradict (e.g. "Only UAE" keeps a prior turn's
metrics/domain and adds a country filter; "Actually use 2025" replaces just
the time range).

The merge itself is the LLM's job, not procedural Python here — this module's
job is building a prompt that gives the model everything it needs to do that
correctly (the full prior slot state, serialized plainly, plus the same
canonical-id catalog app/semantic/router.py uses) and validating the
structured result. Downstream, app/semantic/resolver.py.SchemaResolver is
still the deterministic backstop that resolves canonical ids to physical
schema or fails safe — exactly as it is for a fresh question — so a
hallucinated id here is no more dangerous than one from the router.
"""
from pydantic import BaseModel

from .. import llm
from ..observability.tracing import llm_call_span
from ..prompts import merge as merge_prompts
from ..semantic.resolver import SemanticLayer
from .state import AgentState


class TimeRange(BaseModel):
    """The time condition exactly as the user expressed it (e.g. "last
    quarter", "in 2025") — kept as free text, same reasoning as
    `llm.FilterSpec.value`, since downstream SQL generation reads time
    conditions straight off question text (see synthesize_question below),
    never off a normalized structured value."""
    description: str


class MergedIntent(BaseModel):
    """Structured output of `merge_intent`. Same shape/rules as
    `llm.RoutedIntent` (see app/semantic/router.py's system prompt for what
    counts as a filter vs. a time condition vs. ambiguity) plus `time_range`,
    which a single-turn question has no need to separate out but a follow-up
    like "actually use 2025" does."""
    domains: list[str]
    metrics: list[str]
    dimensions: list[str]
    filters: list[llm.FilterSpec]
    time_range: TimeRange | None
    needs_clarification: bool
    clarification_reason: str
    interpretations: list[llm.Interpretation]
    # True for ordinary ambiguity (genuinely asking which of several things
    # is meant — needs_clarification=true, interpretations populated) or an
    # imprecise-but-answerable question (resolved normally). False only when
    # the message isn't a legitimate data request at all — a destructive
    # command, something with no possible mapping onto the catalog — in
    # which case needs_clarification is still true but interpretations
    # stays EMPTY: there's no choice to offer, so the UI must not present
    # one (see app/agent/graph.py's merge_intent_node and the "refused"
    # status this distinction drives). Defaults true so every existing
    # caller/test constructing a MergedIntent for the ordinary case is
    # unaffected.
    answerable: bool = True


def merge_intent(
    prior_state: AgentState,
    user_message: str,
    semantic_layer: SemanticLayer,
    parse_structured=llm._parse_structured,
    pending_interpretations: list[llm.Interpretation] | None = None,
) -> MergedIntent:
    """Classify `user_message` against `semantic_layer`'s catalog, merged
    with `prior_state`. `parse_structured` defaults to app.llm._parse_structured
    and is injectable — same DI pattern as app/semantic/router.py's `route` —
    so this is unit-testable with a fake, no live LLM API key required.
    `pending_interpretations` is only passed by app/agent/graph.py's
    clarify_node — the interpretations offered on the turn this message is
    replying to, so the same classification call can resolve a reply that
    selects one, dismisses the question entirely ("proceed anyway"), or
    raises a new ambiguity, instead of re-running blind with no memory of
    what was actually asked."""
    prompts = merge_prompts.active()
    system = prompts.system_prompt(semantic_layer, prior_state, pending_interpretations)
    with llm_call_span("merge_intent", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, user_message, MergedIntent, 1000)
    return out


def synthesize_question(state: AgentState) -> str:
    """Deterministic, no LLM: renders resolved slot state back into a plain
    natural-language question string, e.g. metrics=["revenue"],
    dimensions=["country"], filters=[{"dimension": "country", "operator":
    "=", "value": "UAE"}], time_range={"description": "in 2025"} ->
    "Show revenue by country for UAE in 2025".

    This matters because app/text2sql/generator.py's SQL generation and the
    back-translation/sanity-check steps in app/validation read conditions
    (especially time) directly off question text, not off structured
    filters/time_range — so the merged slot state has to be turned back into
    text before it's handed to text2sql_pipeline.run(), which is exactly what
    this function does.
    """
    metrics = state.get("metrics") or []
    dimensions = state.get("dimensions") or []
    filters = state.get("filters") or []
    time_range = state.get("time_range")

    if metrics:
        verb, subject = "Show", " and ".join(metrics)
    elif dimensions:
        verb, subject = "List", " and ".join(dimensions)
    else:
        verb, subject = "Show", "the data"

    parts = [f"{verb} {subject}"]
    if metrics and dimensions:
        parts.append(f"by {' and '.join(dimensions)}")

    for f in filters:
        if f["operator"] == "=":
            parts.append(f"for {f['value']}")
        else:
            parts.append(f"where {f['dimension']} {f['operator']} {f['value']}")

    if time_range and time_range.get("description"):
        parts.append(time_range["description"])

    return " ".join(parts)
