"""Unit tests for app/agent/graph.py — the LangGraph wiring connecting
app/agent/merge.py to the existing text2sql pipeline. Every injection point
(merge_intent's LLM call, pipeline.run's generate/execute/repair) is faked,
matching tests/test_text2sql/test_pipeline.py's approach, so no live
Postgres/LLM API key is needed and no real network call is ever made.
"""
import asyncio

from app import llm
from app.agent import intent, reformulate
from app.agent.graph import build_graph
from app.agent.merge import MergedIntent
from app.agent.state import new_session
from app.db.connection import SecurityContext
from app.db.executor import ExecutionResult
from app.schema_extract import Table
from app.semantic.value_resolver import ValueResolver

TABLE_NAMES = ["orders", "customers", "order_items", "products", "suppliers", "warehouses"]


def _live_schema():
    return {n: Table(name=n, columns=[], primary_keys=[], foreign_keys=[], row_count=0)
            for n in TABLE_NAMES}


def _fake_generate(question, linked, fewshots):
    return llm.GeneratedSQL(
        needs_clarification=False, interpretations=[], sql="SELECT COUNT(*) AS n FROM orders",
        explanation="test", confidence=0.9, tables_used=["orders"], columns_used=[],
    )


async def _fake_executor(sql):
    return ExecutionResult(ok=True, columns=["n"], rows=[[1]], row_count=1)


async def _unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
    raise AssertionError("repair should not have been called")


def _merge_fake(merged_intent: MergedIntent):
    def fake_parse_structured(system, user, model_cls, max_tokens):
        return merged_intent
    return fake_parse_structured


def _always_data_question(system, user, model_cls, max_tokens):
    """Default fake for classify_message_node's LLM call (app/agent/intent.py)
    — only reached once a turn already has a prior pipeline_result, so any
    multi-turn test here would otherwise fall through to the REAL
    app.llm._parse_structured and make a live LLM call. Every test in this
    file is about the merge/pipeline wiring, not intent classification
    itself (that's tests/test_agent/test_intent.py), so "always a data
    question" keeps them offline and deterministic by default."""
    return intent.MessageIntent(kind="data_question", reasoning="test default")


def _echo_reformulate(system, user, model_cls, max_tokens):
    """Default fake for reformulate_question's LLM call — only reached on a
    second-or-later turn (app/agent/reformulate.py skips it entirely on the
    first). Echoing the input unchanged keeps every existing test's merge
    fake (which already ignores its `user_message` argument and returns a
    fixed MergedIntent regardless) behaving exactly as before, while still
    exercising the reformulation code path deterministically offline."""
    return reformulate.ReformulatedQuestion(question=user)


def _build(merge_parse_structured, max_agent_steps=None,
           intent_parse_structured=None, reformulate_parse_structured=None):
    return build_graph(
        pool=None,
        live_schema=_live_schema(),
        fewshots=[],
        generate_fn=_fake_generate,
        executor_fn=_fake_executor,
        repair_fn=_unused_repair,
        merge_parse_structured=merge_parse_structured,
        intent_parse_structured=intent_parse_structured or _always_data_question,
        reformulate_parse_structured=reformulate_parse_structured or _echo_reformulate,
        max_agent_steps=max_agent_steps,
    )


def _invoke(graph, state, thread_id="t1"):
    return asyncio.run(graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}}))


def test_single_turn_question_reaches_ok_with_real_pipeline_result():
    merged = MergedIntent(
        domains=["sales_performance"], metrics=["order_count"], dimensions=[], filters=[],
        time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
    )
    graph = _build(_merge_fake(merged))

    state = new_session("s1")
    state["user_message"] = "how many orders"
    out = _invoke(graph, state)

    assert out["status"] == "ok"
    assert out["pipeline_result"]["status"] == "ok"
    assert out["pipeline_result"]["row_count"] == 1
    assert out["metrics"] == ["order_count"]


def test_resolved_time_range_pulls_orders_table_and_order_date_into_the_generator_schema():
    """Guards a real bug: merge_intent resolves a `time_range` (e.g. "in
    2025") as free text, but build_routed_intent_from_state used to drop it
    entirely when building the pipeline's RoutedIntent — nothing mapped a
    time condition to any table, so a question like "top products by revenue
    in 2025" never pulled the orders table (and its order_date column) into
    the schema the generator sees, which then had to falsely claim no date
    column existed at all. The catalog's designated time_dimension
    (order_date -> orders) must now be appended automatically whenever
    time_range is set, so orders ends up in `linked.tables` even though
    nothing else about this question's metrics/dimensions would have pulled
    it in."""
    merged = MergedIntent(
        domains=["sales_performance"], metrics=["revenue"],
        dimensions=["product"], filters=[],
        time_range={"description": "in 2025"},
        needs_clarification=False, clarification_reason="", interpretations=[],
    )
    captured_linked = {}

    def capturing_generate(question, linked, fewshots):
        captured_linked["linked"] = linked
        return llm.GeneratedSQL(
            needs_clarification=False, interpretations=[], sql="SELECT 1",
            explanation="test", confidence=0.9, tables_used=[], columns_used=[],
        )

    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=capturing_generate, executor_fn=_fake_executor, repair_fn=_unused_repair,
        merge_parse_structured=_merge_fake(merged),
        intent_parse_structured=_always_data_question,
        reformulate_parse_structured=_echo_reformulate,
    )

    state = new_session("s1")
    state["user_message"] = "top products by revenue in 2025"
    _invoke(graph, state)

    linked = captured_linked["linked"]
    assert "orders" in linked.tables
    assert any(d.name == "order_date" for d in linked.dimensions)


def test_generator_receives_users_literal_wording_not_just_synthesized_slots():
    """Guards a real bug: synthesize_question() only renders what resolved
    SLOTS capture (e.g. "Show order_count by order_channel and order_status")
    — nuance the catalog has no slot for ("rate", "percentage") gets silently
    dropped when that's all the generator sees. The literal user message
    must still reach generation alongside the synthesized summary so a
    rate/ratio phrasing isn't lost before the generator's own rate-handling
    prompt rule (app/text2sql/prompts.py) ever gets a chance to see it."""
    merged = MergedIntent(
        domains=["sales_performance"], metrics=["order_count"],
        dimensions=["order_channel", "order_status"], filters=[],
        time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
    )
    captured_question = {}

    def recording_generate(question, linked, fewshots):
        captured_question["question"] = question
        return llm.GeneratedSQL(
            needs_clarification=False, interpretations=[], sql="SELECT 1",
            explanation="test", confidence=0.9, tables_used=["orders"], columns_used=[],
        )

    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=recording_generate, executor_fn=_fake_executor, repair_fn=_unused_repair,
        merge_parse_structured=_merge_fake(merged),
    )
    state = new_session("s1")
    state["user_message"] = "Cancellation rate by sales channel"
    _invoke(graph, state)

    question = captured_question["question"]
    assert "Cancellation rate by sales channel" in question
    assert "order_count" in question  # synthesized summary is still present too


def test_meta_question_after_data_answer_gets_conversational_reply_not_a_new_query():
    """Guards the real bug this feature closes: a message ABOUT the previous
    answer ("explain that") must not be merged against resolved slots like
    an ordinary follow-up (which would just silently re-ask the same data
    question) — it should route to a plain conversational reply instead,
    without calling merge_intent or the pipeline a second time at all."""
    merged = MergedIntent(
        domains=["sales_performance"], metrics=["order_count"], dimensions=[], filters=[],
        time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
    )

    def classify_as_meta(system, user, model_cls, max_tokens):
        return intent.MessageIntent(kind="meta", reasoning="asking about the prior answer")

    def unused_merge_second_call(system, user, model_cls, max_tokens):
        raise AssertionError("merge_intent must not be called for a meta turn")

    merge_calls = {"n": 0}

    def merge_fake_once(system, user, model_cls, max_tokens):
        merge_calls["n"] += 1
        if merge_calls["n"] > 1:
            return unused_merge_second_call(system, user, model_cls, max_tokens)
        return merged

    def fake_respond(system, user, model_cls, max_tokens):
        return intent.ConversationalReply(reply="I grouped by order status because that's what the data showed.")

    graph = _build(merge_fake_once)
    state = new_session("s1")
    state["user_message"] = "how many orders"
    state = _invoke(graph, state)
    assert state["status"] == "ok"

    # classify_message and respond_conversationally are two different calls
    # sharing one injected `intent_parse_structured` — dispatch on which
    # structured model was requested rather than call order.
    def dispatch(system, user, model_cls, max_tokens):
        if model_cls is intent.MessageIntent:
            return classify_as_meta(system, user, model_cls, max_tokens)
        return fake_respond(system, user, model_cls, max_tokens)

    graph3 = _build(merge_fake_once, intent_parse_structured=dispatch)
    state["user_message"] = "why did you group it that way?"
    state = _invoke(graph3, state, thread_id="t1")

    assert state["status"] == "conversational"
    assert state["last_reply"] == "I grouped by order status because that's what the data showed."
    # The real data answer from turn 1 is untouched, still available for a
    # LATER meta question to reference.
    assert state["pipeline_result"]["status"] == "ok"
    assert state["history"][-2] == {"role": "user", "content": "why did you group it that way?"}
    assert state["history"][-1] == {"role": "assistant", "content": state["last_reply"]}


def test_unanswerable_message_is_refused_without_pausing_or_offering_a_choice():
    """Guards a real bug: a destructive command ("DROP TABLE customers")
    with no legitimate mapping onto the catalog was previously either (a)
    silently substituted with the nearest unrelated metric and presented as
    a normal success, or (b) if it did set needs_clarification, treated
    identically to genuine ambiguity — offering interpretations/"proceed
    anyway" with nothing to actually offer. answerable=false must short-
    circuit straight to a terminal "refused" status: no interrupt pause
    (nothing to wait for a reply about), no interpretations, no pipeline
    call at all."""
    unanswerable = MergedIntent(
        domains=[], metrics=[], dimensions=[], filters=[], time_range=None,
        needs_clarification=True, answerable=False,
        clarification_reason="This system only runs read-only queries and can't execute DROP TABLE.",
        interpretations=[],
    )
    graph = _build(_merge_fake(unanswerable))

    state = new_session("s1")
    state["user_message"] = "DROP TABLE customers"
    out = _invoke(graph, state)

    assert out["status"] == "refused"
    assert "__interrupt__" not in out  # terminal, never paused
    assert out["pending_clarification"]["interpretations"] == []
    assert "DROP TABLE" in out["pending_clarification"]["explanation"]


def test_two_turn_followup_reflects_merged_state_not_a_fresh_interpretation():
    first_merged = MergedIntent(
        domains=["sales_performance"], metrics=["revenue"], dimensions=[], filters=[],
        time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
    )
    graph = _build(_merge_fake(first_merged))

    state = new_session("s1")
    state["user_message"] = "show revenue"
    state = _invoke(graph, state)
    assert state["metrics"] == ["revenue"]

    # Second turn: "Only UAE" — a correct merge carries forward domain/metrics
    # and adds the country filter, exactly what app/agent/merge.py's own
    # tests simulate; this test checks the GRAPH applies that result to
    # state rather than discarding it for a fresh interpretation.
    second_merged = MergedIntent(
        domains=["sales_performance"], metrics=["revenue"], dimensions=[],
        filters=[llm.FilterSpec(dimension="customer_country", operator="=", value="UAE")],
        time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
    )
    graph2 = _build(_merge_fake(second_merged))
    state["user_message"] = "Only UAE"
    state = _invoke(graph2, state, thread_id="t1")

    assert state["metrics"] == ["revenue"]  # carried forward, not dropped
    assert state["filters"] == [{"dimension": "customer_country", "operator": "=", "value": "UAE"}]
    assert state["pipeline_result"]["metrics"] == ["revenue"]


def test_agent_steps_cap_stops_instead_of_looping_forever():
    ambiguous = MergedIntent(
        domains=[], metrics=[], dimensions=[], filters=[], time_range=None,
        needs_clarification=True, clarification_reason="always ambiguous", interpretations=[],
    )
    graph = _build(_merge_fake(ambiguous), max_agent_steps=3)

    state = new_session("s1")
    for i in range(5):
        state["user_message"] = f"turn {i}"
        state = _invoke(graph, state, thread_id="t-cap")
        if state["status"] == "error":
            break

    assert state["status"] == "error"
    assert state["agent_steps"] == 4  # 3 ambiguous turns + the one that trips the cap
    assert "too many" in state["pipeline_result"]["error"]


def test_security_context_from_invoke_config_reaches_pipeline_unmodified():
    """The non-negotiable constraint this graph must never violate: the
    agent/LLM never constructs or decides permissions/region — the real
    SecurityContext from the authenticated request (passed in via
    `config["configurable"]["security_context"]`, same as
    app/routes/query.py does) must reach app/text2sql/pipeline.py exactly as
    given, with no node in between able to substitute or drop it. Proven
    here via value grounding (app/semantic/value_resolver.py), the one place
    in the pipeline a custom injected dependency actually receives
    security_context as an argument."""
    captured = {}

    async def fake_lookup_fn(pool, dimension, value, security_context, limit):
        captured["security_context"] = security_context
        return [value]  # pretend the DB found an exact match

    # customer_segment has no curated value_aliases in layer.yaml, so
    # ValueResolver.resolve() falls through to this DB-lookup tier instead
    # of resolving the value from the alias map alone (which never touches
    # security_context at all). A falsy pool short-circuits that tier
    # before lookup_fn is ever called (see value_resolver.py), so this needs
    # a dummy truthy one even though lookup_fn itself is faked and never
    # actually uses it.
    value_resolver = ValueResolver(pool=object(), lookup_fn=fake_lookup_fn)

    merged = MergedIntent(
        domains=["customer_analytics"], metrics=[], dimensions=["customer_segment"],
        filters=[llm.FilterSpec(dimension="customer_segment", operator="=", value="enterprise")],
        time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
    )
    graph = build_graph(
        pool=None,
        live_schema=_live_schema(),
        fewshots=[],
        generate_fn=_fake_generate,
        executor_fn=_fake_executor,
        repair_fn=_unused_repair,
        value_resolver=value_resolver,
        merge_parse_structured=_merge_fake(merged),
    )

    real_security_context = SecurityContext(
        username="alice", role="sales_us", region="us", permissions=frozenset({"query:execute"}),
    )
    state = new_session("s1")
    state["user_message"] = "enterprise customers"
    out = asyncio.run(graph.ainvoke(
        state,
        config={
            "configurable": {
                "thread_id": "t-security",
                "security_context": real_security_context,
            },
        },
    ))

    assert out["status"] == "ok"
    # Identity, not just equality — proves the exact object the request
    # authenticated travelled through merge_intent_node/run_pipeline_node
    # untouched, rather than the agent layer building its own.
    assert captured["security_context"] is real_security_context
