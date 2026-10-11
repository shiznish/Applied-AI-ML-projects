"""Execution-accuracy evaluation package.

`evaluator.py` runs a golden question through app/text2sql/pipeline.py and
scores it against a reference query by executing both and comparing result
sets (app/validation/validation.py:results_agree) — execution accuracy, not
SQL string match. `metrics.py` rolls per-case outcomes up into per-category
accuracy with exact Clopper-Pearson confidence intervals.

Both modules take their pipeline/executor calls as injectable functions so
they're unit-testable with canned fakes — no live Postgres instance or LLM
API key required (see tests/test_evaluation/).
"""
