from app.agent.state import AgentState, append_history, new_session


def test_new_session_has_expected_defaults():
    state = new_session("abc123")

    assert state["session_id"] == "abc123"
    assert state["user_message"] == ""
    assert state["history"] == []
    assert state["history_summary"] is None
    assert state["domain"] is None
    assert state["metrics"] == []
    assert state["dimensions"] == []
    assert state["filters"] == []
    assert state["time_range"] is None
    assert state["pending_clarification"] is None
    assert state["agent_steps"] == 0
    assert state["repair_attempts"] == 0
    assert state["status"] == "new"
    assert state["pipeline_result"] is None
    assert state["last_reply"] is None


def test_agent_state_accepts_all_declared_fields():
    state: AgentState = {
        "session_id": "s1",
        "user_message": "show revenue",
        "history": [{"role": "user", "content": "show revenue"}],
        "history_summary": None,
        "domain": "sales",
        "metrics": ["revenue"],
        "dimensions": ["country"],
        "filters": [{"dimension": "country", "operator": "=", "value": "UAE"}],
        "time_range": {"relative": "last_quarter"},
        "pending_clarification": None,
        "agent_steps": 1,
        "repair_attempts": 0,
        "status": "ok",
        "pipeline_result": {"status": "ok"},
        "last_reply": None,
    }

    assert set(state.keys()) == set(AgentState.__annotations__.keys())


def test_append_history_adds_entry_without_mutating_original():
    original = [{"role": "user", "content": "first"}]
    updated = append_history(original, "assistant", "a reply")

    assert original == [{"role": "user", "content": "first"}]  # not mutated
    assert updated == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a reply"},
    ]


def test_append_history_never_truncates():
    """The full transcript must always make it into agent_sessions.state_jsonb
    (see app/agent/session_store.py) — budgeting what a given prompt renders
    is app/agent/memory.py's job, not append_history's. This replaces the old
    cap-at-MAX_HISTORY_MESSAGES test now that the cap is gone."""
    history: list[dict] = []
    total = 30  # comfortably past the old 20-message cap
    for i in range(total):
        history = append_history(history, "user", f"message {i}")

    assert len(history) == total
    assert history[0]["content"] == "message 0"
    assert history[-1]["content"] == f"message {total - 1}"
