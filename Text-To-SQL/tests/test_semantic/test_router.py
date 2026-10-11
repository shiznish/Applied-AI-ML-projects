"""Unit tests for app/semantic/router.py.

The router's own structured-output call is injectable (`parse_structured`,
same pattern as app/text2sql/generator.py) so these tests fake the LLM call
entirely — no live API key needed — and instead verify: the catalog prompt
actually includes every domain/metric/dimension (so a real model has enough
to work with), and that `route()` passes the question through correctly and
returns whatever RoutedIntent the injected fake produced.

What these tests deliberately do NOT verify: that a real LLM correctly
classifies a given question. That's an accuracy question, answered by
eval/run_eval.py's live-model run against the golden dataset, not a unit
test — offline unit tests and the live eval suite have different jobs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import llm
from app.prompts import catalog as catalog_prompts
from app.semantic import router
from app.semantic.resolver import get as get_layer


def test_catalog_prompt_includes_every_domain_metric_and_dimension():
    layer = get_layer()
    prompt = catalog_prompts.active().catalog_prompt(layer)
    for name in layer.domains:
        assert name in prompt
    for name in layer.metrics:
        assert name in prompt
    for name in layer.dimensions:
        assert name in prompt


def test_catalog_prompt_surfaces_curated_synonyms():
    """A synonym declared in layer.yaml must actually reach the prompt text
    — otherwise curating it was pointless."""
    layer = get_layer()
    prompt = catalog_prompts.active().catalog_prompt(layer)
    assert "customer spend" in prompt  # revenue's curated synonym


def test_route_passes_question_through_to_parse_structured():
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        captured["user"] = user
        captured["model_cls"] = model_cls
        return llm.RoutedIntent(
            domains=["customer_analytics"], metrics=["revenue"], dimensions=["customer_country"],
            filters=[], needs_clarification=False, clarification_reason="", interpretations=[],
        )

    result = router.route("top customers by spend", get_layer(), parse_structured=fake_parse_structured)

    assert captured["user"] == "top customers by spend"
    assert captured["model_cls"] is llm.RoutedIntent
    assert result.metrics == ["revenue"]


def test_route_returns_needs_clarification_when_the_fake_says_so():
    def fake_parse_structured(system, user, model_cls, max_tokens):
        return llm.RoutedIntent(
            domains=[], metrics=[], dimensions=[], filters=[], needs_clarification=True,
            clarification_reason="ambiguous between revenue and campaign_spend",
            interpretations=[
                llm.Interpretation(description="Customer purchase revenue", example_sql=""),
                llm.Interpretation(description="Marketing campaign budget", example_sql=""),
            ],
        )

    result = router.route("top spenders", get_layer(), parse_structured=fake_parse_structured)
    assert result.needs_clarification is True
    assert len(result.interpretations) == 2
