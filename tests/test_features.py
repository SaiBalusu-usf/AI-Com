from cricket_commentary.data.features import compute_features
from cricket_commentary.data.ingest import BallRecord

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


def _record(**kw):
    base = dict(
        match_id="m", innings=1, over=1, ball=1, is_legal=True,
        batting_team="A", bowling_team="B", batter="Bat", bowler="Bowl",
        non_striker="NS", runs_batter=0, runs_total=0, extras=0,
        extras_type="none", wicket=False, wicket_type="", player_out="",
        team_score=0, team_wickets=0, target=None, balls_remaining=119,
        balls_bowled=1, innings_balls_total=120,
    )
    base.update(kw)
    return BallRecord(**base)


def test_run_rates():
    # 30 runs off 24 legal balls -> CRR 7.5; chase of 100 with 70 needed off 96
    rec = _record(team_score=30, balls_bowled=24, target=100, balls_remaining=96)
    feats = compute_features([rec], FEATURES_CFG)[0]
    assert feats["current_run_rate"] == 7.5
    assert feats["required_run_rate"] == round(70 / 96 * 6, 2)


def test_rrr_none_outside_chase_and_zero_when_won():
    assert compute_features([_record()], FEATURES_CFG)[0]["required_run_rate"] is None
    won = _record(team_score=101, target=100, balls_remaining=10, balls_bowled=110)
    assert compute_features([won], FEATURES_CFG)[0]["required_run_rate"] == 0.0


def test_phase_boundaries():
    for over, expected in [(1, "powerplay"), (6, "powerplay"), (7, "middle"),
                           (15, "middle"), (16, "death"), (20, "death")]:
        feats = compute_features([_record(over=over)], FEATURES_CFG)[0]
        assert feats["phase"] == expected, over


def test_salience_table():
    cases = [
        (dict(wicket=True, wicket_type="bowled"), 1.0),
        (dict(runs_batter=6, runs_total=6), 0.8),
        (dict(runs_batter=4, runs_total=4), 0.6),
        (dict(over=17), 0.4),                          # dot in the death
        (dict(runs_batter=2, runs_total=2), 0.2),      # else runs/10
        (dict(), 0.0),                                 # plain dot mid-powerplay
    ]
    for overrides, expected in cases:
        feats = compute_features([_record(**overrides)], FEATURES_CFG)[0]
        assert feats["event_salience"] == expected, overrides


def test_momentum_window_excludes_current_ball():
    records = [
        _record(ball=i % 6 + 1, over=i // 6 + 1, runs_total=2, balls_bowled=i + 1)
        for i in range(14)
    ]
    feats = compute_features(records, FEATURES_CFG)
    assert feats[0]["momentum"] == 0           # nothing before the first ball
    assert feats[1]["momentum"] == 2
    assert feats[13]["momentum"] == 24         # capped at the 12-ball window


def test_momentum_resets_between_innings_and_matches():
    a = _record(runs_total=6)
    b = _record(innings=2, runs_total=6)
    c = _record(match_id="other", runs_total=6)
    feats = compute_features([a, b, c], FEATURES_CFG)
    assert [f["momentum"] for f in feats] == [0, 0, 0]


def test_tension_bounded_and_rises_with_pressure():
    calm = _record(target=180, team_score=100, balls_bowled=60, balls_remaining=60,
                   team_wickets=1)
    tight = _record(target=180, team_score=150, balls_bowled=108, balls_remaining=12,
                    team_wickets=7, over=19)
    feats = compute_features([calm, tight], FEATURES_CFG)
    assert 0.0 <= feats[0]["tension"] <= 1.0
    assert 0.0 <= feats[1]["tension"] <= 1.0
    assert feats[1]["tension"] > feats[0]["tension"]


def test_first_innings_tension_uses_phase_and_salience():
    quiet = _record(over=8)                                   # middle over dot
    big = _record(over=19, wicket=True, wicket_type="caught") # death wicket
    feats = compute_features([quiet, big], FEATURES_CFG)
    assert feats[0]["tension"] == round(0.5 * 0.35 + 0.5 * 0.0, 3)
    assert feats[1]["tension"] == round(0.5 * 0.9 + 0.5 * 1.0, 3)
