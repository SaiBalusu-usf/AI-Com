"""End-to-end harness test on the committed fixture with the dummy
copy-the-input generator — the Phase 2 gate (§8)."""

import json

import pytest

from cricket_commentary.eval.harness import evaluate
from cricket_commentary.utils.config import load_config


@pytest.fixture(scope="module")
def fixture_rows():
    path = "data/fixtures/mini.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def eval_cfg(request):
    return load_config("configs/eval.yaml")


def _copy_input(row: dict) -> str:
    return row["linearized_input"]


def test_harness_end_to_end_on_fixture(fixture_rows, eval_cfg):
    generations = [_copy_input(r) for r in fixture_rows]
    metrics = evaluate(fixture_rows, generations, eval_cfg, generate_fn=_copy_input)

    assert set(metrics) == {"faithfulness", "excitement", "surface", "diversity", "latency"}
    faith = metrics["faithfulness"]
    assert faith["n"] == 200
    assert 0.0 <= faith["hallucination_rate"] <= 1.0
    # the linearized input restates the structured facts (score=X/Y etc. is
    # not claim-shaped English), so the copy generator should hallucinate ~nothing
    assert faith["hallucination_rate"] < 0.1

    # scorer validation ran on references first and must pass on the fixture
    validation = metrics["excitement"]["reference_validation"]
    assert validation["passed"] is True and validation["spearman_rho"] > 0
    assert "spearman_rho" in metrics["excitement"]["calibration"]

    assert 0 <= metrics["surface"]["bleu"] <= 100
    assert metrics["surface"]["bert_score"].get("status") in ("unavailable", "disabled", None) or (
        "f1" in metrics["surface"]["bert_score"]
    )
    assert metrics["diversity"]["n_generations"] == 200
    assert metrics["latency"]["n_timed"] == 200
    assert metrics["latency"]["p50_ms"] <= metrics["latency"]["p95_ms"]


def test_harness_alignment_guards(fixture_rows, eval_cfg):
    with pytest.raises(ValueError):
        evaluate(fixture_rows, ["x"], eval_cfg)
    with pytest.raises(ValueError):
        evaluate([], [], eval_cfg)


def test_eval_only_run_records_latency_gap(fixture_rows, eval_cfg):
    rows = fixture_rows[:20]
    metrics = evaluate(rows, [r["target_commentary"] for r in rows], eval_cfg)
    assert metrics["latency"]["status"] == "unavailable"
