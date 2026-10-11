"""Unit tests for app/agent/reformulate.py."""
from app.agent.reformulate import ReformulatedQuestion, reformulate_question
from app.agent.state import new_session
from app.prompts.safety import CONVERSATION_SAFETY_PREAMBLE


def test_skips_llm_call_with_no_prior_history():
    def unused(system, user, model_cls, max_tokens):
        raise AssertionError("must not call the LLM with nothing to reformulate against")

    state = new_session("s1")
    state["history"] = [{"role": "user", "content": "less order value"}]  # just this turn's own
    result = reformulate_question(state, "less order value", parse_structured=unused)

    assert result == "less order value"


def test_reformulates_using_prior_context_and_includes_safety_preamble():
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        return ReformulatedQuestion(question="Which country has the lowest average order value?")

    state = new_session("s1")
    state["history"] = [
        {"role": "user", "content": "Which country has the highest average order value?"},
        {"role": "assistant", "content": "Returned 1 row(s)."},
        {"role": "user", "content": "less order value"},
    ]
    result = reformulate_question(state, "less order value", parse_structured=fake_parse_structured)

    assert result == "Which country has the lowest average order value?"
    assert "highest average order value" in captured["system"]
    assert CONVERSATION_SAFETY_PREAMBLE in captured["system"]


def test_prompt_defaults_to_unchanged_and_warns_against_softening_explicit_statements():
    """Guards a real bug: a later message that already names its own subject
    explicitly ("DROP TABLE customers") got genericized into a vague "Show
    the data" — losing the specific entity it named and leaving semantic
    routing with nothing concrete to resolve, which made the generator
    hallucinate a placeholder table name. The prompt must tell the model to
    default to UNCHANGED and never soften/drop a word a complete, explicit
    message specifically named."""
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        return ReformulatedQuestion(question=user)

    state = new_session("s1")
    state["history"] = [
        {"role": "user", "content": "how many orders"},
        {"role": "assistant", "content": "Returned 1 row(s)."},
    ]
    reformulate_question(state, "DROP TABLE customers", parse_structured=fake_parse_structured)

    system = captured["system"].lower()
    assert "default to returning the message unchanged" in system
    assert "do not soften, generalize" in system
