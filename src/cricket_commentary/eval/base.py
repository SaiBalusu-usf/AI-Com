"""Shared plumbing for the evaluation harness (§5).

Metric availability policy (§2 rule 7): a metric that cannot be computed is
NEVER silently skipped and never faked. Modules raise
:class:`MetricUnavailable`; the harness records
``{"status": "unavailable", "reason": ...}`` in metrics.json and logs loudly.
"""

from __future__ import annotations

import numpy as np


class MetricUnavailable(Exception):
    """Raised when a metric's dependency (library, model, network) is absent."""


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def _average_ranks(values: list[float] | np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (1-based), scipy-free."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y) -> float | None:
    """Spearman rank correlation with average-rank ties.

    Returns None when undefined (n < 3 or a constant series) — callers must
    surface that as a real gap, not as 0.0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("spearman: length mismatch")
    if len(x) < 3:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def group_by_innings(rows: list[dict]) -> dict[tuple[str, int], list[int]]:
    """Row indices grouped by (match_id, innings), preserving ball order."""
    groups: dict[tuple[str, int], list[int]] = {}
    for i, row in enumerate(rows):
        rec = row["record"]
        groups.setdefault((rec["match_id"], rec["innings"]), []).append(i)
    return groups
