"""Unit tests for app/agent/merge.py.

merge_intent's own merging logic is entirely delegated to the LLM via the
prompt it builds (see merge.py's module docstring) — these tests fake that
call, so what they can actually verify is: (1) the prior slot state is
correctly serialized into the prompt a real LLM would see, and (2) whatever
structured MergedIntent the model (or, here, the fake standing in for it)
returns passes through merge_intent unchanged. Whether a real LLM correctly
performs the merge is an accuracy question for eval/run_eval.py, not this
unit test.
"""
from app import llm
from app.agent.merge import MergedIntent, TimeRange, merge_intent, synthesize_question
from app.agent.state import new_session
from app.semantic.resolver import get as get_layer


def test_fresh_message_with_no_prior_state_produces_full_intent():
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        captured["user"] = user
        return MergedIntent(
            domains=["sales_performance"],
            metrics=["revenue"],
            dimensions=["customer_country"],
            filters=[],
            time_range=None,
            needs_clarification=False,
            clarification_reason="",
            interpretations=[],
        )

    prior = new_session("s1")
    result = merge_intent(
        prior, "show revenue by country", get_layer(), parse_structured=fake_parse_structured
    )

    assert captured["user"] == "show revenue by country"
    assert "- domain: none" in captured["system"]
    assert "- metrics: none" in captured["system"]
    assert result.metrics == ["revenue"]
    assert result.dimensions == ["customer_country"]


def test_merged_intent_defaults_answerable_true():
    """Every existing caller/fake constructs a MergedIntent without
    `answerable` — it must default to True so none of them break now that
    the field exists (see app/agent/merge.py's MergedIntent docstring for
    what answerable=False means and why it needs its own distinct outcome,
    not just another flavor of needs_clarification)."""
    merged = MergedIntent(
        domains=[], metrics=[], dimensions=[], filters=[], time_range=None,
        needs_clarification=False, clarification_reason="", interpretations=[],
    )
    assert merged.answerable is True


def test_system_prompt_instructs_refusing_unanswerable_requests():
    """Guards the fix for a real bug: a destructive command with no
    legitimate mapping onto the catalog was previously silently resolved to
    the nearest unrelated metric and presented as a normal success. The
    prompt must tell the model to set answerable=false and leave
    interpretations EMPTY for this case, distinct from ordinary ambiguity."""
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        return MergedIntent(
            domains=[], metrics=[], dimensions=[], filters=[], time_range=None,
            needs_clarification=True, answerable=False,
            clarification_reason="not something this system can do",
            interpretations=[],
        )

    prior = new_session("s1")
    result = merge_intent(
        prior, "DROP TABLE customers", get_layer(), parse_structured=fake_parse_structured
    )

    assert "answerable=false" in captured["system"]
    assert "interpretations must stay empty" in captured["system"].lower()
    assert result.answerable is False
    assert result.interpretations == []


def test_system_prompt_warns_against_carrying_time_range_into_an_unrelated_question():
    """Guards a real bug caught via live use: after "Top 5 products by
    revenue in 2025", the unrelated follow-up "Which country has the highest
    average order value?" (a different metric AND dimension, no continuation
    language) silently inherited time_range="in 2025" from the prior turn —
    the old prompt's only guard against this was a vague "if genuinely new
    and self-contained, treat as fresh" rule, and the model judged the two
    revenue-ish questions as related. The prompt must give a concrete,
    name-the-exact-case rule tying time_range/filter carry-forward to
    actually continuing the SAME metric, not just a nearby business area —
    this checks the prompt wording itself (the model's actual judgment isn't
    testable offline), the same way the other system-prompt tests in this
    file do."""
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        return MergedIntent(
            domains=["customer_analytics"], metrics=["average_order_value"],
            dimensions=["customer_country"], filters=[], time_range=None,
            needs_clarification=False, clarification_reason="", interpretations=[],
        )

    prior = new_session("s1")
    prior["domain"] = "sales_performance"
    prior["metrics"] = ["revenue"]
    prior["dimensions"] = ["product"]
    prior["time_range"] = {"description": "in 2025"}
    merge_intent(
        prior, "Which country has the highest average order value?", get_layer(),
        parse_structured=fake_parse_structured,
    )

    system = captured["system"].lower()
    assert "different metric" in system
    assert "time_range=none" in system or "fresh, self-contained" in system


def test_follow_up_only_uae_carries_forward_metrics_and_domain():
    prior = new_session("s1")
    prior["domain"] = "sales_performance"
    prior["metrics"] = ["revenue"]

    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        # Simulates what a real LLM should produce for this follow-up:
        # prior metrics/domain untouched, new country filter added.
        return MergedIntent(
            domains=["sales_performance"],
            metrics=["revenue"],
            dimensions=[],
            filters=[llm.FilterSpec(dimension="customer_country", operator="=", value="UAE")],
            time_range=None,
            needs_clarification=False,
            clarification_reason="",
            interpretations=[],
        )

    result = merge_intent(prior, "Only UAE", get_layer(), parse_structured=fake_parse_structured)

    # The prior state must actually reach the prompt, or a real LLM would
    # have nothing to carry forward.
    assert "- metrics: revenue" in captured["system"]
    assert "- domain: sales_performance" in captured["system"]
    assert result.metrics == ["revenue"]
    assert result.domains == ["sales_performance"]
    assert len(result.filters) == 1
    assert result.filters[0].value == "UAE"


def test_follow_up_actually_use_2025_only_changes_time_range():
    prior = new_session("s1")
    prior["domain"] = "sales_performance"
    prior["metrics"] = ["revenue"]
    prior["filters"] = [{"dimension": "customer_country", "operator": "=", "value": "UAE"}]

    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        return MergedIntent(
            domains=["sales_performance"],
            metrics=["revenue"],
            dimensions=[],
            filters=[llm.FilterSpec(dimension="customer_country", operator="=", value="UAE")],
            time_range=TimeRange(description="in 2025"),
            needs_clarification=False,
            clarification_reason="",
            interpretations=[],
        )

    result = merge_intent(
        prior, "Actually use 2025", get_layer(), parse_structured=fake_parse_structured
    )

    assert "- filters: customer_country = UAE" in captured["system"]
    assert result.metrics == ["revenue"]
    assert result.filters[0].value == "UAE"
    assert result.time_range.description == "in 2025"


def test_ambiguous_followup_sets_needs_clarification():
    prior = new_session("s1")

    def fake_parse_structured(system, user, model_cls, max_tokens):
        return MergedIntent(
            domains=[],
            metrics=[],
            dimensions=[],
            filters=[],
            time_range=None,
            needs_clarification=True,
            clarification_reason="ambiguous between revenue and campaign_spend",
            interpretations=[
                llm.Interpretation(description="Customer purchase revenue", example_sql=""),
                llm.Interpretation(description="Marketing campaign budget", example_sql=""),
            ],
        )

    result = merge_intent(
        prior, "top spend", get_layer(), parse_structured=fake_parse_structured
    )

    assert result.needs_clarification is True
    assert len(result.interpretations) == 2


def test_synthesize_question_renders_metrics_dimensions_filters_and_time_range():
    state = new_session("s1")
    state["metrics"] = ["revenue"]
    state["dimensions"] = ["customer_country"]
    state["filters"] = [{"dimension": "customer_country", "operator": "=", "value": "UAE"}]
    state["time_range"] = {"description": "in 2025"}

    assert synthesize_question(state) == "Show revenue by customer_country for UAE in 2025"


def test_synthesize_question_with_only_dimensions_uses_list_verb():
    state = new_session("s1")
    state["dimensions"] = ["product_category"]

    assert synthesize_question(state) == "List product_category"


def test_synthesize_question_with_non_equality_filter():
    state = new_session("s1")
    state["metrics"] = ["order_count"]
    state["filters"] = [{"dimension": "order_status", "operator": "!=", "value": "cancelled"}]

    assert synthesize_question(state) == "Show order_count where order_status != cancelled"


def test_synthesize_question_with_nothing_resolved():
    state = new_session("s1")
    assert synthesize_question(state) == "Show the data"
