"""Fine-tuning tests. The pure-python pieces (formatting, device planning)
run everywhere; actual training is exercised by
``scripts/train_model.py --smoke`` on machines with the ML stack
(needs_torch), because the build environment cannot install torch (PLAN.md D2).
"""

import importlib.util
import json

import pytest

from cricket_commentary.models.fewshot import GeneratorUnavailable
from cricket_commentary.models.finetune import (
    build_inference,
    format_chat,
    format_seq2seq,
    resolve_device_plan,
    train,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def row():
    with open("data/fixtures/mini.jsonl") as f:
        return json.loads(f.readline())


def test_format_seq2seq(row):
    src, tgt = format_seq2seq(row)
    assert src.startswith("describe this ball: over=")
    assert row["linearized_input"] in src
    assert tgt == row["target_commentary"]


def test_format_chat(row):
    messages = format_chat(row, "You are a live cricket commentator. Describe this ball.")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"] == row["linearized_input"]
    assert messages[2]["content"] == row["target_commentary"]


@pytest.mark.skipif(HAS_TORCH, reason="ML stack present; unavailability contract n/a")
def test_device_plan_reports_missing_stack():
    plan = resolve_device_plan({"model": {"quantize_4bit": True}})
    assert plan["available"] is False
    assert "setup-full" in plan["reason"]


@pytest.mark.skipif(HAS_TORCH, reason="ML stack present; unavailability contract n/a")
def test_train_and_inference_fail_loudly_without_stack(row):
    with pytest.raises(GeneratorUnavailable, match="setup-full"):
        train({"system": "finetune_seq2seq", "model": {}}, [row], [row], "/tmp/x")
    with pytest.raises(GeneratorUnavailable, match="setup-full"):
        build_inference({"system": "finetune_seq2seq", "model": {}})


@pytest.mark.needs_torch
@pytest.mark.skipif(not HAS_TORCH, reason="needs make setup-full")
def test_device_plan_with_stack():
    plan = resolve_device_plan({"model": {"quantize_4bit": True}})
    assert plan["available"] is True
    assert isinstance(plan["use_4bit"], bool)
    # on non-CUDA machines the degradation must be explicit, never silent
    if not plan["cuda"]:
        assert plan["use_4bit"] is False and plan["note"]
