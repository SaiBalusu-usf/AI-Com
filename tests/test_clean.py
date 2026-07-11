from cricket_commentary.data.clean import clean_targets, entity_mismatch, strip_html
from cricket_commentary.data.ingest import BallRecord


def _record(**kw):
    base = dict(
        match_id="m", innings=1, over=1, ball=1, is_legal=True,
        batting_team="A", bowling_team="B", batter="V Kohli", bowler="R Khan",
        non_striker="S Gill", runs_batter=1, runs_total=1, extras=0,
        extras_type="none", wicket=False, wicket_type="", player_out="",
        team_score=1, team_wickets=0, target=None, balls_remaining=119,
        balls_bowled=1, innings_balls_total=120,
        squad=["V Kohli", "R Khan", "S Gill", "M Sharma", "T Boult"],
    )
    base.update(kw)
    return BallRecord(**base)


def test_strip_html():
    assert strip_html("<b>Four!</b>&nbsp;lovely   shot") == "Four! lovely shot"


def test_entity_mismatch_flags_wrong_actor():
    assert entity_mismatch("Sharma drives crisply for one", _record())
    assert not entity_mismatch("Kohli drives crisply for one", _record())
    # naming a non-actor alongside the real actor is fine (e.g. captain chat)
    assert not entity_mismatch("Kohli single; Sharma applauds", _record())
    # generic text without any squad name is not a mismatch
    assert not entity_mismatch("Tidy single into the covers", _record())


def test_clean_targets_reasons_are_counted():
    rows = [
        {"target_commentary": "<i>Kohli</i> tucks it away for a single"},
        {"target_commentary": "ok"},                                # too short
        {"target_commentary": "word " * 90},                        # too long
        {"target_commentary": "Sharma hammers it down the ground"}, # misaligned
        {"target_commentary": "Kohli tucks it away for a single"},  # dup of row 0 post-clean
    ]
    records = [_record() for _ in rows]
    kept, stats = clean_targets(rows, records, min_tokens=4, max_tokens=80)
    assert len(kept) == 1
    assert stats["dropped_too_short"] == 1
    assert stats["dropped_too_long"] == 1
    assert stats["dropped_entity_mismatch"] == 1
    assert stats["dropped_duplicate"] == 1
    assert kept[0]["target_commentary"] == "Kohli tucks it away for a single"
