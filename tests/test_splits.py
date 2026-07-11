"""§2 rule 5: splits are by match, never by ball. These tests are the
enforcement demanded by the build spec."""

import pytest

from cricket_commentary.data.splits import split_by_match

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_no_match_spans_two_splits():
    # simulate ball-level input: every match id appears many times
    match_ids = [f"m{i}" for i in range(20) for _ in range(120)]
    assignment = split_by_match(match_ids, FRACTIONS, seed=13)
    # one assignment per match — by construction a match cannot leak
    assert set(assignment) == {f"m{i}" for i in range(20)}
    balls = [(mid, assignment[mid]) for mid in match_ids]
    per_match_splits = {}
    for mid, split in balls:
        per_match_splits.setdefault(mid, set()).add(split)
    assert all(len(s) == 1 for s in per_match_splits.values())


def test_all_splits_nonempty_and_proportional():
    assignment = split_by_match([f"m{i}" for i in range(20)], FRACTIONS, seed=13)
    counts = {name: list(assignment.values()).count(name) for name in FRACTIONS}
    assert counts["train"] == 16 and counts["val"] == 2 and counts["test"] == 2


def test_small_match_counts_still_cover_every_split():
    assignment = split_by_match(["a", "b", "c", "d", "e", "f"], FRACTIONS, seed=13)
    assert set(assignment.values()) == {"train", "val", "test"}
    assert len(assignment) == 6


def test_deterministic_and_seed_sensitive():
    ids = [f"m{i}" for i in range(30)]
    assert split_by_match(ids, FRACTIONS, seed=7) == split_by_match(ids, FRACTIONS, seed=7)
    assert split_by_match(ids, FRACTIONS, seed=7) != split_by_match(ids, FRACTIONS, seed=8)


def test_bad_fractions_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        split_by_match(["a"], {"train": 0.5, "test": 0.1}, seed=1)
    with pytest.raises(ValueError, match="no matches"):
        split_by_match([], FRACTIONS, seed=1)
