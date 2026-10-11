"""Unit tests for app/agent/intent.py — the data-question vs. meta classifier
and the conversational-reply builder. Both are exercised through the graph in
tests/test_agent/test_graph.py's meta-routing test; these focus on the
prompt content itself (the actual LLM call is always faked — no live API key
needed, matching every other test in this package)."""
from app.agent.intent import classify_message, respond_conversationally
from app.agent.state import new_session
from app.prompts.safety import CONVERSATION_SAFETY_PREAMBLE


def _state_with_prior_answer():
    state = new_session("s1")
    state["history"] = [
        {"role": "user", "content": "Show revenue by country"},
        {"role": "assistant", "content": "Returned 5 row(s). Revenue grouped by country."},
    ]
    state["pipeline_result"] = {
        "status": "ok", "sql": "SELECT country, SUM(revenue) FROM orders GROUP BY country",
        "metrics": ["revenue"], "dimensions": ["customer_country"], "row_count": 5,
        "violations": [],
    }
    return state


def test_classify_message_prompt_includes_transcript_and_last_answer():
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        captured["user"] = user
        from app.agent.intent import MessageIntent
        return MessageIntent(kind="meta", reasoning="test")

    state = _state_with_prior_answer()
    result = classify_message(state, "why did you group by country?", parse_structured=fake_parse_structured)

    assert result.kind == "meta"
    assert captured["user"] == "why did you group by country?"
    assert "Show revenue by country" in captured["system"]  # transcript included
    assert "SELECT country, SUM(revenue)" in captured["system"]  # last answer included


def test_classify_message_with_no_history_still_works():
    def fake_parse_structured(system, user, model_cls, max_tokens):
        from app.agent.intent import MessageIntent
        return MessageIntent(kind="data_question", reasoning="test")

    state = new_session("s1")
    result = classify_message(state, "how many orders", parse_structured=fake_parse_structured)

    assert result.kind == "data_question"


def test_classify_message_prompt_includes_safety_preamble():
    """respond_conversationally's output is shown to the user verbatim with
    nothing downstream to catch a manipulated answer the way
    app/guardrails/sql_validator.py catches manipulated SQL — this and the
    other new LLM calls this feature added (classify_message,
    reformulate_question) all paste the full raw conversation transcript
    into their own system prompt, so each needs its own prompt-injection
    framing instead of relying on the SQL-path backstop that doesn't apply
    to them. Guards against that framing being silently dropped later."""
    def fake_parse_structured(system, user, model_cls, max_tokens):
        from app.agent.intent import MessageIntent
        assert CONVERSATION_SAFETY_PREAMBLE in system
        return MessageIntent(kind="data_question", reasoning="test")

    classify_message(new_session("s1"), "how many orders", parse_structured=fake_parse_structured)


def test_respond_conversationally_returns_reply_text_grounded_in_context():
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        from app.agent.intent import ConversationalReply
        return ConversationalReply(reply="I grouped by country because that's what you asked for.")

    state = _state_with_prior_answer()
    reply = respond_conversationally(state, "why country?", parse_structured=fake_parse_structured)

    assert reply == "I grouped by country because that's what you asked for."
    assert "customer_country" in captured["system"]
    assert CONVERSATION_SAFETY_PREAMBLE in captured["system"]
    assert "never reveal" in captured["system"].lower()
