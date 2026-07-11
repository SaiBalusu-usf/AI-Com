"""Train/val/test assignment BY MATCH, never by ball (§2 rule 5).

Splitting by ball leaks near-duplicate events across splits; the unit tests
in tests/test_splits.py enforce that no match ever spans two splits.
"""

from __future__ import annotations

import numpy as np


def split_by_match(
    match_ids: list[str], fractions: dict[str, float], seed: int
) -> dict[str, str]:
    """Deterministic {match_id: split_name}. Fractions apply to match counts;
    every split with a positive fraction gets at least one match when enough
    matches exist."""
    total = sum(fractions.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    unique = sorted(set(match_ids))
    if not unique:
        raise ValueError("no matches to split")

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique))

    names = list(fractions)
    counts = {name: int(len(shuffled) * fractions[name]) for name in names}
    # ensure non-empty splits, then hand leftovers to train-like biggest split
    for name in names:
        if fractions[name] > 0 and counts[name] == 0 and len(shuffled) >= len(names):
            counts[name] = 1
    while sum(counts.values()) > len(shuffled):
        counts[max(counts, key=counts.get)] -= 1
    counts[max(names, key=lambda n: fractions[n])] += len(shuffled) - sum(counts.values())

    assignment: dict[str, str] = {}
    cursor = 0
    for name in names:
        for match_id in shuffled[cursor : cursor + counts[name]]:
            assignment[match_id] = name
        cursor += counts[name]
    return assignment
