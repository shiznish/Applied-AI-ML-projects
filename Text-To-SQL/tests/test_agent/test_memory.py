"""Unit tests for app/agent/memory.py — the budgeted long-term-memory layer.

These tests fake the summarization LLM call (same DI pattern as every other
module in this layer) so they never need a live API key, and instead verify
the pure budget bookkeeping (`build_context`) plus the node's behavior
around when it does/doesn't call the LLM and what it does/doesn't change in
state.
"""
from app.agent.memory import ConversationSummary, build_context, estimate_tokens, render_transcript, summarize_node
from app.agent.state import append_history, new_session


def _history(n: int, content_len: int = 20) -> list[dict]:
    history: list[dict] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history = append_history(history, role, f"message {i} " + ("x" * content_len))
    return history


def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_roughly_quarters_length():
    text = "a" * 40
    assert estimate_tokens(text) == 10


def test_build_context_under_budget_returns_unchanged():
    history = _history(4)
    first, recent = build_context(history, "old summary", token_budget=10_000)

    assert first == "old summary"
    assert recent == history  # unchanged, not even a copy requirement


def test_build_context_over_budget_splits_into_older_and_recent():
    history = _history(30, content_len=50)  # comfortably large
    older, recent = build_context(history, None, token_budget=200)

    assert isinstance(older, list)
    assert older + recent == history  # chronological order preserved, nothing dropped
    assert len(older) > 0
    assert len(recent) > 0
    # The recent tail must actually fit the budget on its own.
    rendered = "\n".join(f'{h["role"]}: {h["content"]}' for h in recent)
    assert estimate_tokens(rendered) <= 200


def test_build_context_empty_history():
    first, recent = build_context([], "summary", token_budget=10)
    assert first == "summary"
    assert recent == []


def test_render_transcript_with_no_history_and_no_summary():
    assert render_transcript(None, []) == "No prior conversation this session."


def test_render_transcript_renders_summary_leading_line_when_present():
    history = _history(2)
    text = render_transcript("Earlier the user asked about revenue.", history)
    assert text.startswith("Summary of earlier conversation: Earlier the user asked about revenue.")
    assert "Conversation so far, oldest first:" in text


def test_summarize_node_is_a_no_op_under_budget():
    """The overwhelmingly common case: the transcript still fits, so no LLM
    call happens at all and no state update is produced."""
    state = new_session("s1")
    state["history"] = _history(4)

    def must_not_call(system, user, model_cls, max_tokens):
        raise AssertionError("must not call the LLM when under budget")

    result = summarize_node(state, parse_structured=must_not_call)
    assert result == {}


def test_summarize_node_folds_oldest_turns_and_updates_summary_only():
    """Over budget: summarize_node must call the injected fake, update
    history_summary, and must NOT touch `history` itself — the full
    transcript has to stay intact for app/agent/session_store.py to persist,
    regardless of what gets folded for prompt-rendering purposes."""
    state = new_session("s1")
    full_history = _history(30, content_len=50)
    state["history"] = full_history
    state["history_summary"] = None

    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        captured["user"] = user
        return ConversationSummary(summary="folded summary of older turns")

    import app.config as config_module
    old_budget = config_module.settings.agent_history_token_budget
    config_module.settings.agent_history_token_budget = 200
    try:
        result = summarize_node(state, parse_structured=fake_parse_structured)
    finally:
        config_module.settings.agent_history_token_budget = old_budget

    assert result == {"history_summary": "folded summary of older turns"}
    assert "none yet" in captured["system"]  # no previous summary existed yet
    # Full history untouched in the state dict passed in.
    assert state["history"] == full_history


def test_summarize_node_passes_previous_summary_forward():
    state = new_session("s1")
    state["history"] = _history(30, content_len=50)
    state["history_summary"] = "prior running summary"

    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        return ConversationSummary(summary="updated summary")

    import app.config as config_module
    old_budget = config_module.settings.agent_history_token_budget
    config_module.settings.agent_history_token_budget = 200
    try:
        result = summarize_node(state, parse_structured=fake_parse_structured)
    finally:
        config_module.settings.agent_history_token_budget = old_budget

    assert result == {"history_summary": "updated summary"}
    assert "prior running summary" in captured["system"]
