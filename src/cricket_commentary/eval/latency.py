"""Latency measurement (§5.3, RQ3): p50/p95 wall-clock per ball and
tokens/sec at batch size 1 after warmup.

Each generation call is timed individually with ``time.perf_counter``;
warmup calls run first and are excluded from the statistics. Hardware
context (platform, CPU count, torch device) is recorded alongside the
numbers so runs on different machines are never compared blindly.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Callable

import numpy as np


def _torch_device() -> str:
    """Best available torch device, or "not_installed" (guarded import —
    torch is an optional dependency of the harness)."""
    try:
        import torch
    except ImportError:
        return "not_installed"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_device": _torch_device(),
    }


def measure_latency(
    generate_fn: Callable[[dict], str], rows: list[dict], cfg: dict
) -> dict:
    """Per-ball latency stats for ``generate_fn`` over ``rows`` (§5.3, RQ3).

    The first ``min(cfg["warmup_calls"], len(rows))`` rows are generated
    without timing (JIT/cache warmup), then every row is generated once
    with each call timed individually. ``tokens_per_sec`` counts
    whitespace tokens across all timed outputs over total timed seconds.
    """
    if not rows:
        raise ValueError("measure_latency: rows is empty")
    if int(cfg.get("batch_size", 1)) != 1:
        raise ValueError("measure_latency: only batch_size=1 is supported (§5.3)")

    warmup_calls = int(cfg["warmup_calls"])
    if warmup_calls < 0:
        raise ValueError("measure_latency: warmup_calls must be >= 0")
    for row in rows[: min(warmup_calls, len(rows))]:
        generate_fn(row)

    durations_s: list[float] = []
    total_tokens = 0
    for row in rows:
        start = time.perf_counter()
        output = generate_fn(row)
        durations_s.append(time.perf_counter() - start)
        total_tokens += len(output.split())

    arr = np.asarray(durations_s, dtype=float)
    total_s = max(float(arr.sum()), 1e-9)
    return {
        "n_timed": len(rows),
        "p50_ms": float(np.percentile(arr, 50)) * 1000.0,
        "p95_ms": float(np.percentile(arr, 95)) * 1000.0,
        "mean_ms": float(arr.mean()) * 1000.0,
        "tokens_per_sec": total_tokens / total_s,
        "hardware": _hardware(),
    }
