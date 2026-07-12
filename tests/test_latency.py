"""Latency module tests (§5.3, RQ3). Structure/ordering assertions only —
no sleep-based timing, which is flaky under CI load."""

from __future__ import annotations

import importlib.util
from typing import Callable

import pytest

from cricket_commentary.eval.latency import measure_latency

CFG = {"warmup_calls": 5, "batch_size": 1}


def _counting_generator() -> tuple[Callable[[dict], str], dict]:
    """Deterministic fake generator (5 whitespace tokens) plus call counter."""
    calls = {"n": 0}

    def generate(row: dict) -> str:
        calls["n"] += 1
        return "one two three four five"

    return generate, calls


def test_counts_and_stat_ordering():
    generate, calls = _counting_generator()
    rows = [{"ball": i} for i in range(12)]
    result = measure_latency(generate, rows, CFG)

    assert result["n_timed"] == len(rows)
    assert calls["n"] == len(rows) + min(CFG["warmup_calls"], len(rows))
    assert result["p50_ms"] <= result["p95_ms"]
    assert result["mean_ms"] > 0
    assert result["tokens_per_sec"] > 0


def test_hardware_block():
    generate, _ = _counting_generator()
    result = measure_latency(generate, [{"ball": 0}], CFG)
    hardware = result["hardware"]

    assert set(hardware) == {"platform", "cpu_count", "torch_device"}
    if importlib.util.find_spec("torch") is None:
        assert hardware["torch_device"] == "not_installed"
    else:  # torch present on this machine: device must still be a known value
        assert hardware["torch_device"] in {"cuda", "mps", "cpu"}


def test_empty_rows_raises():
    generate, _ = _counting_generator()
    with pytest.raises(ValueError):
        measure_latency(generate, [], CFG)


def test_fewer_rows_than_warmup_calls():
    generate, calls = _counting_generator()
    rows = [{"ball": i} for i in range(2)]  # 2 rows < 5 warmup_calls
    result = measure_latency(generate, rows, CFG)

    assert result["n_timed"] == 2
    assert calls["n"] == 2 + 2  # warmup clamped to len(rows)
