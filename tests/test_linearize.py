from cricket_commentary.data.ingest import BallRecord
from cricket_commentary.data.linearize import linearize, outcome_label


def _record(**kw):
    base = dict(
        match_id="m", innings=2, over=18, ball=3, is_legal=True,
        batting_team="A", bowling_team="B", batter="Kohli", bowler="Rashid Khan",
        non_striker="NS", runs_batter=6, runs_total=6, extras=0,
        extras_type="none", wicket=False, wicket_type="", player_out="",
        team_score=162, team_wickets=4, target=195, balls_remaining=15,
        balls_bowled=105, innings_balls_total=120,
    )
    base.update(kw)
    return BallRecord(**base)


FEATS = {
    "required_run_rate": 9.8,
    "wickets_in_hand": 6,
    "phase": "death",
    "momentum": 13,
    "event_salience": 0.8,
    "tension": 0.7,
}


def test_flat_format_matches_spec_shape():
    line = linearize(_record(), FEATS, fmt="flat", include_game_state=True)
    assert line == (
        "over=17.3 | batter=Kohli | bowler=Rashid Khan | outcome=SIX | runs=6 "
        "| score=162/4 | RRR=9.8 | target=195 | wkts_in_hand=6 | phase=death "
        "| momentum=13"
    )


def test_game_state_toggle_is_clean():
    line = linearize(_record(), FEATS, include_game_state=False)
    for field in ("RRR=", "wkts_in_hand=", "phase=", "momentum=", "target="):
        assert field not in line
    assert "outcome=SIX" in line and "score=162/4" in line


def test_context_format_prefixes_previous_outcomes():
    line = linearize(
        _record(), FEATS, fmt="context", prev_outcomes=["SIX", "FOUR", "DOT"],
        context_balls=2,
    )
    assert line.startswith("prev=FOUR,DOT | over=17.3")  # last N only
    first = linearize(_record(), FEATS, fmt="context", prev_outcomes=[])
    assert first.startswith("prev=NONE | ")


def test_outcome_labels():
    assert outcome_label(_record(wicket=True, wicket_type="run out")) == "WICKET(RUN_OUT)"
    assert outcome_label(_record(runs_batter=4, runs_total=4)) == "FOUR"
    assert outcome_label(_record(runs_batter=0, runs_total=0)) == "DOT"
    assert outcome_label(_record(runs_batter=1, runs_total=1)) == "SINGLE"
    assert (
        outcome_label(_record(runs_batter=0, runs_total=1, extras=1, extras_type="wides"))
        == "WIDES1"
    )
    # wicket dominates other facts in the label
    assert outcome_label(
        _record(wicket=True, wicket_type="caught", runs_batter=0, runs_total=0)
    ) == "WICKET(CAUGHT)"


def test_wicket_adds_player_out_slot():
    rec = _record(wicket=True, wicket_type="bowled", player_out="Kohli",
                  runs_batter=0, runs_total=0)
    line = linearize(rec, FEATS)
    assert "player_out=Kohli" in line and "outcome=WICKET(BOWLED)" in line
