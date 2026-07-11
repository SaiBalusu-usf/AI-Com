"""Contract tests for the COMMITTED fixture — the golden path every phase
gate and `make mini` runs on. If someone regenerates it badly, this fails."""

import json

REQUIRED_ROW_KEYS = {
    "record", "features", "linearized_input", "target_commentary",
    "split", "data_source", "reference_source",
}
REQUIRED_RECORD_KEYS = {
    "match_id", "innings", "over", "ball", "batter", "bowler", "non_striker",
    "runs_batter", "extras", "extras_type", "wicket", "wicket_type",
    "player_out", "team_score", "team_wickets", "target", "balls_remaining",
}
REQUIRED_FEATURE_KEYS = {
    "current_run_rate", "required_run_rate", "wickets_in_hand",
    "balls_remaining", "phase", "momentum", "event_salience", "tension",
}


def _rows(fixture_path):
    with open(fixture_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fixture_has_200_schema_complete_rows(fixture_path):
    rows = _rows(fixture_path)
    assert len(rows) == 200
    for row in rows:
        assert REQUIRED_ROW_KEYS <= set(row)
        assert REQUIRED_RECORD_KEYS <= set(row["record"])
        assert REQUIRED_FEATURE_KEYS <= set(row["features"])
        assert 4 <= len(row["target_commentary"].split()) <= 80


def test_fixture_is_labelled_synthetic(fixture_path):
    for row in _rows(fixture_path):
        assert row["data_source"] == "synthetic"
        assert row["reference_source"] == "synthetic_v1"


def test_fixture_split_by_match_and_all_splits_present(fixture_path):
    rows = _rows(fixture_path)
    per_match = {}
    for row in rows:
        per_match.setdefault(row["record"]["match_id"], set()).add(row["split"])
    assert all(len(s) == 1 for s in per_match.values()), "match spans splits!"
    assert {s for splits in per_match.values() for s in splits} == {"train", "val", "test"}


def test_fixture_covers_all_event_types(fixture_path):
    rows = _rows(fixture_path)
    records = [r["record"] for r in rows]
    assert any(r["wicket"] for r in records)
    assert any(r["runs_batter"] == 6 for r in records)
    assert any(r["runs_batter"] == 4 for r in records)
    assert any(r["extras_type"] != "none" for r in records)
    assert any(r["runs_total"] == 0 for r in records)
    assert any(r["target"] is not None for r in records)  # chase balls
    assert any(r["wicket"] and r["wicket_type"] == "caught" for r in records)


def test_fixture_tension_and_salience_in_range(fixture_path):
    for row in _rows(fixture_path):
        assert 0.0 <= row["features"]["tension"] <= 1.0
        assert 0.0 <= row["features"]["event_salience"] <= 1.0
