"""The simulator must emit schema-perfect Cricsheet JSON: `ingest.parse_match`
is the single parser for real and synthetic data, so these tests double as a
schema contract."""

from cricket_commentary.data.features import compute_features
from cricket_commentary.data.ingest import parse_match
from cricket_commentary.data.synthetic import simulate_match, write_reference

FEATURES_CFG = {
    "momentum_window_balls": 12,
    "phases": {"powerplay": [1, 6], "middle": [7, 15], "death": [16, 20]},
    "salience": {"wicket": 1.0, "six": 0.8, "four": 0.6, "dot_in_death": 0.4},
    "tension": {
        "chase": {"rrr_weight": 0.4, "wickets_weight": 0.3, "progress_weight": 0.3, "rrr_scale": 2.0},
        "first_innings": {
            "phase_weight": 0.5,
            "salience_weight": 0.5,
            "phase_weights": {"powerplay": 0.5, "middle": 0.35, "death": 0.9},
        },
    },
}


def test_deterministic_given_seed():
    assert simulate_match(99, 3) == simulate_match(99, 3)
    assert simulate_match(99, 3) != simulate_match(99, 4)


def test_parses_through_the_real_ingest_path():
    match_id, match = simulate_match(13, 0)
    records = parse_match(match, match_id)
    assert len(records) > 120  # two innings of a T20

    # cumulative score must equal the sum of delivery totals per innings
    for innings_no, innings in enumerate(match["innings"], start=1):
        recs = [r for r in records if r.innings == innings_no]
        total = sum(
            d["runs"]["total"] for o in innings["overs"] for d in o["deliveries"]
        )
        assert recs[-1].team_score == total


def test_overs_have_exactly_six_legal_balls_except_terminal():
    match_id, match = simulate_match(21, 1)
    for innings in match["innings"]:
        overs = innings["overs"]
        for i, over_block in enumerate(overs):
            legal = sum(
                1
                for d in over_block["deliveries"]
                if not set((d.get("extras") or {})) & {"wides", "noballs"}
            )
            if i < len(overs) - 1:
                assert legal == 6, f"non-final over with {legal} legal balls"
            else:
                assert 1 <= legal <= 6


def test_chase_stops_at_target_or_before():
    for match_no in range(4):
        match_id, match = simulate_match(7, match_no)
        second = match["innings"][1]
        target = second["target"]["runs"]
        runs = sum(d["runs"]["total"] for o in second["overs"] for d in o["deliveries"])
        overshoot = runs - target
        assert overshoot <= 6, "chase kept batting after winning"


def test_squads_are_distinct_and_full():
    _, match = simulate_match(5, 2)
    players = match["info"]["players"]
    teams = match["info"]["teams"]
    assert set(players) == set(teams)
    all_names = [p for squad in players.values() for p in squad]
    assert len(all_names) == 22
    assert len({name.split()[-1] for name in all_names}) == 22  # unique surnames


def test_strike_rotation_follows_the_laws():
    """Within an over: wides rotate on runs-run (extras minus the 1-run
    penalty), no-balls on batter runs, byes/legbyes on runs taken, plain
    deliveries on batter runs. Caught in the phase-7 review: v1 rotated on
    total runs for wides/no-balls (inverted vs the laws of cricket)."""
    for match_no in range(6):
        match_id, match = simulate_match(31, match_no)
        for innings in match["innings"]:
            for over_block in innings["overs"]:
                expected = None  # unknown at over start and after a wicket
                for d in over_block["deliveries"]:
                    if expected is not None:
                        assert d["batter"] == expected, (
                            f"{match_id} over {over_block['over']}: expected "
                            f"{expected} on strike, got {d['batter']}"
                        )
                    if d.get("wickets"):
                        expected = None
                        continue
                    extras = d.get("extras") or {}
                    if "wides" in extras:
                        ran = d["runs"]["extras"] - 1
                    elif "noballs" in extras:
                        ran = d["runs"]["batter"]
                    elif extras:
                        ran = d["runs"]["extras"]
                    else:
                        ran = d["runs"]["batter"]
                    expected = d["non_striker"] if ran % 2 == 1 else d["batter"]


def test_delivery_seq_is_unique_per_over():
    match_id, match = simulate_match(13, 0)
    records = parse_match(match, match_id)
    keys = [(r.match_id, r.innings, r.over, r.delivery_seq) for r in records]
    assert len(keys) == len(set(keys)), "delivery_seq must uniquely key deliveries"
    # and the old (over, ball) key genuinely collides for wides, proving the need
    ball_keys = [(r.match_id, r.innings, r.over, r.ball) for r in records]
    assert len(ball_keys) > len(set(ball_keys))


def test_references_are_deterministic_and_entity_faithful():
    match_id, match = simulate_match(13, 0)
    records = parse_match(match, match_id)
    feats = compute_features(records, FEATURES_CFG)
    a = write_reference(records[10], feats[10], seed=13)
    b = write_reference(records[10], feats[10], seed=13)
    assert a == b and len(a.split()) >= 3
    # wicket references name the dismissed batter or the bowler
    for rec, ft in zip(records, feats):
        if rec.wicket:
            text = write_reference(rec, ft, seed=13)
            surnames = {rec.batter.split()[-1], rec.bowler.split()[-1]}
            assert any(s in text for s in surnames)
