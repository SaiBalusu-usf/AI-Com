"""Hand-choreographed Cricsheet-format match exercising cumulative-state math:
legal-ball counting around wides, post-ball scores, wickets, batter tallies,
maiden stamping, target plumbing, and super-over skipping."""

import pytest

from cricket_commentary.data.ingest import over_ball_display, parse_match


def _delivery(batter, bowler, runs_batter=0, extras=0, total=None, extras_field=None, wickets=None):
    d = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "N Striker",
        "runs": {
            "batter": runs_batter,
            "extras": extras,
            "total": runs_batter + extras if total is None else total,
        },
    }
    if extras_field:
        d["extras"] = extras_field
    if wickets:
        d["wickets"] = wickets
    return d


@pytest.fixture()
def match():
    over0 = [
        _delivery("A One", "X Bowler", runs_batter=1),
        _delivery("B Two", "X Bowler", extras=1, extras_field={"wides": 1}),
        _delivery("B Two", "X Bowler", runs_batter=4),
        _delivery("B Two", "X Bowler", wickets=[{"kind": "bowled", "player_out": "B Two"}]),
        _delivery("C Three", "X Bowler", runs_batter=6),
        _delivery("C Three", "X Bowler"),
        _delivery("C Three", "X Bowler", extras=2, extras_field={"byes": 2}),
    ]
    return {
        "info": {
            "match_type": "T20",
            "overs": 20,
            "teams": ["Alpha", "Beta"],
            "players": {
                "Alpha": ["A One", "B Two", "C Three", "N Striker"],
                "Beta": ["X Bowler", "Y Fielder"],
            },
        },
        "innings": [
            {"team": "Alpha", "overs": [{"over": 0, "deliveries": over0}]},
            {
                "team": "Beta",
                "target": {"runs": 15, "overs": 20},
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [_delivery("X Bowler", "A One", runs_batter=4)],
                    }
                ],
            },
            {"team": "Alpha", "super_over": True, "overs": []},
        ],
    }


def test_cumulative_state_through_first_over(match):
    records = parse_match(match, "m1")
    first = [r for r in records if r.innings == 1]
    assert len(first) == 7

    # legal-ball numbering: the wide carries the upcoming legal ball's number
    assert [r.ball for r in first] == [1, 2, 2, 3, 4, 5, 6]
    assert [r.is_legal for r in first] == [True, False, True, True, True, True, True]

    # post-ball team score: 1, +1wd, +4, +0(W), +6, +0, +2byes
    assert [r.team_score for r in first] == [1, 2, 6, 6, 12, 12, 14]
    assert [r.team_wickets for r in first] == [0, 0, 0, 1, 1, 1, 1]

    wicket_ball = first[3]
    assert wicket_ball.wicket and wicket_ball.wicket_type == "bowled"
    assert wicket_ball.player_out == "B Two"

    # batter personal tally after each ball he faced (4 off the bat, then out)
    assert first[2].batter_runs_after == 4
    assert first[2].batter_balls_faced == 1  # the wide did not count as a ball faced

    # balls_remaining counts only legal deliveries
    assert first[0].balls_remaining == 119
    assert first[1].balls_remaining == 119  # unchanged by the wide
    assert first[-1].balls_remaining == 114

    # over completed with 6 legal balls: total over runs stamped on final ball
    assert first[-1].completed_over_runs == 14
    assert all(r.completed_over_runs is None for r in first[:-1])


def test_extras_typing_and_totals(match):
    records = parse_match(match, "m1")
    wide, bye = records[1], records[6]
    assert wide.extras_type == "wides" and wide.extras == 1 and wide.runs_total == 1
    assert bye.extras_type == "byes" and bye.runs_total == 2 and bye.runs_batter == 0


def test_second_innings_target_and_teams(match):
    records = parse_match(match, "m1")
    chase = [r for r in records if r.innings == 2]
    assert len(chase) == 1
    assert chase[0].target == 15
    assert chase[0].batting_team == "Beta" and chase[0].bowling_team == "Alpha"
    assert [r.target for r in records if r.innings == 1] == [None] * 7


def test_super_over_skipped(match):
    records = parse_match(match, "m1")
    assert {r.innings for r in records} == {1, 2}


def test_over_ball_display_uses_scoreboard_notation(match):
    records = parse_match(match, "m1")
    assert over_ball_display(records[0]) == "0.1"
    assert over_ball_display(records[6]) == "0.6"  # last ball of the first over


def test_squad_collected_for_misattribution_checks(match):
    records = parse_match(match, "m1")
    assert "Y Fielder" in records[0].squad and "A One" in records[0].squad


def test_two_teams_required():
    with pytest.raises(ValueError, match="2 teams"):
        parse_match({"info": {"teams": ["OnlyOne"]}, "innings": []}, "bad")
