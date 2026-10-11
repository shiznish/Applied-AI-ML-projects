"""Offline unit tests for app/evaluation/evaluator.py. Both the pipeline call
and the reference-SQL executor are injected as fakes (pipeline_run_fn,
executor_fn), following the same dependency-injection pattern used in
tests/test_text2sql/ — no live Postgres instance or LLM API key needed.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.executor import ExecutionResult
from app.evaluation.evaluator import evaluate_case, evaluate_cases
from app.text2sql.pipeline import PipelineResult


def _ok_result(rows, columns):
    return ExecutionResult(ok=True, columns=columns, rows=rows, row_count=len(rows))


def _pipeline_fn(pipeline_result):
    async def fake(question, live_schema, pool, fewshots, semantic_layer=None, security_context=None):
        return pipeline_result
    return fake


def _executor_fn(execution_result):
    async def fake(pool, sql, security_context=None):
        return execution_result
    return fake


NORMAL_CASE = {
    "id": "t01", "category": "simple", "difficulty": "easy",
    "question": "How many customers do we have?",
    "expected_sql": "SELECT COUNT(*) FROM customers",
}

AMBIGUOUS_CASE = {
    "id": "t02", "category": "ambiguous", "difficulty": "medium",
    "question": "What is our total sales?",
    "expect_ambiguous": True,
}


def test_matching_result_set_passes():
    generated = _ok_result([[5]], ["n"])
    reference = _ok_result([[5]], ["n"])
    pr = PipelineResult(status="ok", sql="SELECT COUNT(*) AS n FROM customers", execution=generated)

    outcome = asyncio.run(evaluate_case(
        NORMAL_CASE, pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=_pipeline_fn(pr), executor_fn=_executor_fn(reference),
    ))

    assert outcome.passed is True
    assert outcome.id == "t01"
    assert outcome.category == "simple"
    assert outcome.status == "ok"
    assert outcome.reason == ""


def test_mismatching_result_set_fails():
    generated = _ok_result([[5]], ["n"])
    reference = _ok_result([[6]], ["n"])
    pr = PipelineResult(status="ok", sql="SELECT COUNT(*) AS n FROM customers", execution=generated)

    outcome = asyncio.run(evaluate_case(
        NORMAL_CASE, pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=_pipeline_fn(pr), executor_fn=_executor_fn(reference),
    ))

    assert outcome.passed is False
    assert "mismatch" in outcome.reason


def test_non_ok_pipeline_status_fails_normal_case():
    pr = PipelineResult(status="blocked", sql="DROP TABLE customers", violations=["DDL not allowed"])

    outcome = asyncio.run(evaluate_case(
        NORMAL_CASE, pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=_pipeline_fn(pr), executor_fn=_executor_fn(_ok_result([[5]], ["n"])),
    ))

    assert outcome.passed is False
    assert "blocked" in outcome.reason


def test_reference_sql_failure_fails_the_case():
    generated = _ok_result([[5]], ["n"])
    bad_reference = ExecutionResult(ok=False, error="syntax error", error_code="42601")
    pr = PipelineResult(status="ok", sql="SELECT COUNT(*) AS n FROM customers", execution=generated)

    outcome = asyncio.run(evaluate_case(
        NORMAL_CASE, pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=_pipeline_fn(pr), executor_fn=_executor_fn(bad_reference),
    ))

    assert outcome.passed is False
    assert "reference" in outcome.reason


def test_ambiguous_case_passes_when_pipeline_reports_ambiguous():
    pr = PipelineResult(status="ambiguous")

    outcome = asyncio.run(evaluate_case(
        AMBIGUOUS_CASE, pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=_pipeline_fn(pr), executor_fn=_executor_fn(_ok_result([], [])),
    ))

    assert outcome.passed is True
    assert outcome.status == "ambiguous"


def test_ambiguous_case_fails_when_pipeline_answers_directly():
    generated = _ok_result([[100]], ["revenue"])
    pr = PipelineResult(status="ok", sql="SELECT SUM(x) AS revenue FROM t", execution=generated)

    outcome = asyncio.run(evaluate_case(
        AMBIGUOUS_CASE, pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=_pipeline_fn(pr), executor_fn=_executor_fn(_ok_result([], [])),
    ))

    assert outcome.passed is False
    assert "ambiguous" in outcome.reason


def test_evaluate_cases_runs_every_case_and_preserves_order():
    ok_pr = PipelineResult(status="ambiguous")

    async def fake_run(question, live_schema, pool, fewshots, semantic_layer=None, security_context=None):
        return ok_pr

    outcomes = asyncio.run(evaluate_cases(
        [AMBIGUOUS_CASE, AMBIGUOUS_CASE], pool=None, live_schema={}, fewshots=[],
        pipeline_run_fn=fake_run, executor_fn=_executor_fn(_ok_result([], [])),
    ))

    assert len(outcomes) == 2
    assert all(o.passed for o in outcomes)
    assert [o.id for o in outcomes] == ["t02", "t02"]
