"""Tests for app/llm.py's secret redaction: _redact() itself, and every
provider error path that builds an LLMError/LLMAuthError message from a
caught SDK exception. A provider auth failure commonly echoes the offending
key back in its message (e.g. "Incorrect API key provided: gsk_..."), and
app/main.py surfaces str(exception) straight into a 502 HTTPException body —
so every one of these construction sites must redact before raising, not
just the one path that historically did.
"""
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import llm

FAKE_KEY = "gsk_abc123fakekeyfortesting"


def _http_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid/v1/x")
    return httpx.Response(status_code, request=request, json={"error": {"message": "x"}})


# --------------------------------------------------------------------------
# _redact() itself
# --------------------------------------------------------------------------

def test_redact_scrubs_groq_style_key():
    assert llm._redact(f"Incorrect API key provided: {FAKE_KEY}") == \
        "Incorrect API key provided: ***REDACTED***"


def test_redact_scrubs_openai_style_key():
    text = "auth failed for sk-abcdefghij1234567890"
    assert FAKE_KEY not in llm._redact(text)
    assert "***REDACTED***" in llm._redact(text)


def test_redact_leaves_ordinary_text_untouched():
    assert llm._redact("Could not reach the API — DNS failure") == \
        "Could not reach the API — DNS failure"


def test_redact_scrubs_configured_api_key_even_without_recognized_prefix(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "totally-custom-secret-value")
    text = "auth failed: totally-custom-secret-value"
    assert "totally-custom-secret-value" not in llm._redact(text)
    assert "***REDACTED***" in llm._redact(text)


# --------------------------------------------------------------------------
# Anthropic path (~line 108-134): AuthenticationError / APIStatusError
# --------------------------------------------------------------------------

class _FakeAnthropicMessages:
    def __init__(self, to_raise):
        self._to_raise = to_raise

    def parse(self, **kwargs):
        raise self._to_raise


class _FakeAnthropicClient:
    def __init__(self, to_raise):
        self.messages = _FakeAnthropicMessages(to_raise)


def _run_anthropic(monkeypatch, exc):
    import anthropic
    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    llm._clients.pop("anthropic", None)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient(exc))
    try:
        llm._parse_structured("system", "user", llm.BackTranslation, 100)
    finally:
        llm._clients.pop("anthropic", None)


def test_anthropic_authentication_error_is_redacted(monkeypatch):
    import anthropic
    exc = anthropic.AuthenticationError(
        f"Incorrect API key provided: {FAKE_KEY}", response=_http_response(401), body=None,
    )
    with pytest.raises(llm.LLMAuthError) as excinfo:
        _run_anthropic(monkeypatch, exc)
    assert FAKE_KEY not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


def test_anthropic_no_credits_error_is_redacted(monkeypatch):
    import anthropic
    exc = anthropic.APIStatusError(
        f"Your credit balance is too low, key {FAKE_KEY}",
        response=_http_response(400), body=None,
    )
    with pytest.raises(llm.LLMAuthError) as excinfo:
        _run_anthropic(monkeypatch, exc)
    assert FAKE_KEY not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


def test_anthropic_generic_status_error_is_redacted(monkeypatch):
    import anthropic
    exc = anthropic.APIStatusError(
        f"bad request, offending key {FAKE_KEY}",
        response=_http_response(400), body=None,
    )
    with pytest.raises(llm.LLMError) as excinfo:
        _run_anthropic(monkeypatch, exc)
    assert FAKE_KEY not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


# --------------------------------------------------------------------------
# OpenAI-compatible path (~line 189-245): AuthenticationError / APIStatusError
# --------------------------------------------------------------------------

class _FakeOpenAIChatCompletions:
    def __init__(self, to_raise):
        self._to_raise = to_raise

    def create(self, **kwargs):
        raise self._to_raise


class _FakeOpenAIChat:
    def __init__(self, to_raise):
        self.completions = _FakeOpenAIChatCompletions(to_raise)


class _FakeOpenAIClient:
    def __init__(self, to_raise):
        self.chat = _FakeOpenAIChat(to_raise)


def _run_openai(monkeypatch, exc):
    import openai
    monkeypatch.setattr(llm.settings, "llm_provider", "groq")
    llm._clients.pop("openai", None)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeOpenAIClient(exc))
    try:
        llm._parse_structured("system", "user", llm.BackTranslation, 100)
    finally:
        llm._clients.pop("openai", None)


def test_openai_authentication_error_is_redacted(monkeypatch):
    import openai
    exc = openai.AuthenticationError(
        f"Incorrect API key provided: {FAKE_KEY}", response=_http_response(401), body=None,
    )
    with pytest.raises(llm.LLMAuthError) as excinfo:
        _run_openai(monkeypatch, exc)
    assert FAKE_KEY not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


def test_openai_status_error_is_redacted(monkeypatch):
    import openai
    exc = openai.APIStatusError(
        f"rate limited, key {FAKE_KEY}", response=_http_response(429), body=None,
    )
    with pytest.raises(llm.LLMError) as excinfo:
        _run_openai(monkeypatch, exc)
    assert FAKE_KEY not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


def test_openai_connection_error_is_redacted(monkeypatch):
    """Pre-existing redaction path — kept covered so a future refactor can't
    silently drop it while adding the ones above."""
    import openai
    cause = RuntimeError(f"SSL handshake failed for key {FAKE_KEY}")
    exc = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    exc.__cause__ = cause
    with pytest.raises(llm.LLMError) as excinfo:
        _run_openai(monkeypatch, exc)
    assert FAKE_KEY not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)
