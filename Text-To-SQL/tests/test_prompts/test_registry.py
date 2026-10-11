"""Unit tests for app/prompts/registry.py's resolve_version — the shared
version-resolution logic every app/prompts/<family>/__init__.py's active()
calls into (see e.g. app/prompts/intent/__init__.py).

Exercises both the generic resolver (with a small fake `available` dict, so
no real prompt family needs a second version just to test the mechanism)
and one real family end-to-end (app.prompts.intent), to prove the override
actually reaches a real package's active() through app.config.settings.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from app.config import settings
from app.prompts import intent as intent_prompts
from app.prompts.registry import resolve_version


def test_resolve_version_uses_default_when_no_override(monkeypatch):
    monkeypatch.setattr(settings, "prompt_version_overrides", {})
    assert resolve_version("some_family", "v1", {"v1": object()}) == "v1"


def test_resolve_version_honors_override_for_named_family(monkeypatch):
    v1, v2 = object(), object()
    monkeypatch.setattr(settings, "prompt_version_overrides", {"some_family": "v2"})
    assert resolve_version("some_family", "v1", {"v1": v1, "v2": v2}) == "v2"


def test_resolve_version_ignores_override_for_a_different_family(monkeypatch):
    """An override keyed by family name must not leak across families —
    only the exact family name requested is looked up."""
    monkeypatch.setattr(settings, "prompt_version_overrides", {"other_family": "v2"})
    assert resolve_version("some_family", "v1", {"v1": object(), "v2": object()}) == "v1"


def test_resolve_version_raises_on_unknown_version(monkeypatch):
    monkeypatch.setattr(settings, "prompt_version_overrides", {"some_family": "v9"})
    with pytest.raises(ValueError) as exc_info:
        resolve_version("some_family", "v1", {"v1": object()})
    # The error must name both the bad family and the bad version requested
    # — this is the whole point of resolving at call time rather than
    # silently falling back, so a misconfigured override fails loudly.
    assert "some_family" in str(exc_info.value)
    assert "v9" in str(exc_info.value)


def test_intent_active_returns_default_version_with_no_override(monkeypatch):
    # intent's DEFAULT_VERSION was promoted to v2 (app/prompts/intent/v2.py
    # — last_answer_prompt surfaces confidence/hallucination-check/
    # multi-query-agreement, which v1 never saw).
    monkeypatch.setattr(settings, "prompt_version_overrides", {})
    assert intent_prompts.active() is intent_prompts.v2
    assert intent_prompts.active().PROMPT_VERSION == intent_prompts.v2.PROMPT_VERSION


def test_intent_active_honors_env_override(monkeypatch):
    # v1 stays selectable via the override even though it's no longer the
    # default — exactly the pin/rollback path this registry exists for.
    monkeypatch.setattr(settings, "prompt_version_overrides", {"intent": "v1"})
    assert intent_prompts.active() is intent_prompts.v1


def test_intent_active_raises_on_unknown_override_version(monkeypatch):
    monkeypatch.setattr(settings, "prompt_version_overrides", {"intent": "v9"})
    with pytest.raises(ValueError) as exc_info:
        intent_prompts.active()
    assert "intent" in str(exc_info.value)
    assert "v9" in str(exc_info.value)


def test_old_version_stays_selectable_after_a_new_version_is_promoted(monkeypatch):
    """Worked example proving the whole point of this design: adding a v2
    never removes v1 — both stay importable/selectable by version string,
    simulated here with a fake family's _VERSIONS/DEFAULT_VERSION rather
    than mutating a real app/prompts/<family>/ package (those are frozen
    once superseded — see app/prompts/registry.py's module docstring)."""

    class _FakeV1:
        PROMPT_VERSION = "fake-2026.1"

    class _FakeV2:
        PROMPT_VERSION = "fake-2026.2"

    versions = {"v1": _FakeV1, "v2": _FakeV2}
    default_version = "v2"  # promoted: v2 is now the default

    def active():
        version = resolve_version("fake_family", default_version, versions)
        return versions[version]

    # Default behavior now serves v2...
    monkeypatch.setattr(settings, "prompt_version_overrides", {})
    assert active() is _FakeV2

    # ...but v1 is still right there, selectable via the override, unchanged.
    monkeypatch.setattr(settings, "prompt_version_overrides", {"fake_family": "v1"})
    assert active() is _FakeV1
    assert active().PROMPT_VERSION == "fake-2026.1"
