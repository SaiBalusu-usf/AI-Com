"""Adversarial unit tests for the slot-level fact checker (§5.1).

The spec demands >=20 hand-written cases including the classic traps:
"no run" vs "run out", "the fourth ball" vs "a four", "single-handedly",
bowler praise vs dismissal credit. The checker is only credible if these pass.
"""

import pytest

from cricket_commentary.eval.factcheck import (
    check_generation,
    extract_claims,
    factcheck_metrics,
)

SQUAD = ["V Kohli", "R Khan", "S Gill", "M Sharma", "T Boult", "J Archer"]


def _record(**kw):
    base = dict(
        match_id="m", innings=1, over=5, ball=3, is_legal=True,
        batting_team="A", bowling_team="B", batter="V Kohli", bowler="R Khan",
        non_striker="S Gill", runs_batter=0, runs_total=0, extras=0,
        extras_type="none", wicket=False, wicket_type="", player_out="",
        fielders=[], team_score=45, team_wickets=2, target=None,
        balls_remaining=90, balls_bowled=30, innings_balls_total=120,
        batter_runs_after=12, batter_balls_faced=9, completed_over_runs=None,
        squad=SQUAD,
    )
    base.update(kw)
    return base


def _violation_slots(text, record):
    return {v["slot"] for v in check_generation(text, record).violations}


# ---------------------------------------------------------------- runs slot

def test_no_run_on_dot_ball_is_clean():
    assert _violation_slots("No run, defended into the covers.", _record()) == set()


def test_no_run_on_scoring_ball_is_violation():
    rec = _record(runs_batter=2, runs_total=2)
    assert "runs" in _violation_slots("No run there.", rec)


def test_no_run_not_confused_with_run_out():
    # "no run out attempted" must claim neither zero-runs nor a wicket
    claims = extract_claims("They hesitate but there is no run out attempted.")
    assert not any(c.slot == "wicket" for c in claims)


def test_run_out_is_a_wicket_claim():
    rec = _record(wicket=True, wicket_type="run out", player_out="V Kohli")
    assert _violation_slots("Kohli is run out by a direct hit!", rec) == set()


def test_single_on_single_is_clean_and_checked():
    rec = _record(runs_batter=1, runs_total=1)
    assert _violation_slots("Quick single to mid-on.", rec) == set()
    assert "runs" in _violation_slots("Quick single to mid-on.", _record(runs_batter=4, runs_total=4))


def test_single_handedly_is_not_a_runs_claim():
    claims = extract_claims("Kohli is single-handedly keeping them alive")
    assert not any(c.slot == "runs" for c in claims)


def test_digit_runs_checked_against_record():
    rec = _record(runs_batter=3, runs_total=3)
    assert _violation_slots("Three runs taken!", rec) == set()
    assert "runs" in _violation_slots("2 runs added", rec)


def test_needed_runs_is_not_a_scoring_claim():
    claims = extract_claims("18 runs needed off the last over")
    assert not any(c.slot == "runs" for c in claims)


# ------------------------------------------------------------ boundary slot

def test_four_off_the_bat_is_clean():
    rec = _record(runs_batter=4, runs_total=4)
    assert _violation_slots("Cracked away for four!", rec) == set()


def test_the_fourth_ball_is_not_a_boundary_claim():
    claims = extract_claims("The fourth ball of the over is defended.")
    assert not any(c.slot == "boundary" for c in claims)


def test_four_leg_byes_is_extras_not_boundary():
    rec = _record(runs_batter=0, runs_total=4, extras=4, extras_type="legbyes")
    text = "Four leg byes, the umpire signals."
    claims = extract_claims(text)
    assert not any(c.slot == "boundary" for c in claims)
    assert any(c.slot == "extras" and c.value == 4 for c in claims)
    assert _violation_slots(text, rec) == set()


def test_hallucinated_four_on_dot_ball():
    assert "boundary" in _violation_slots("Beautifully driven for four", _record())


def test_six_and_maximum_are_six_claims():
    rec = _record(runs_batter=6, runs_total=6)
    assert _violation_slots("SIX! That is a maximum!", rec) == set()
    assert "boundary" in _violation_slots("A maximum to finish!", _record())


def test_sixth_over_is_not_a_six_claim():
    claims = extract_claims("Into the sixth over now.")
    assert not any(c.slot == "boundary" for c in claims)


def test_out_of_here_is_a_six_idiom_not_a_wicket():
    rec = _record(runs_batter=6, runs_total=6)
    assert _violation_slots("Kohli unloads - that's out of here! Maximum!", rec) == set()


# -------------------------------------------------------------- wicket slot

def test_bowled_dismissal_is_clean():
    rec = _record(wicket=True, wicket_type="bowled", player_out="V Kohli")
    assert _violation_slots("Bowled him! Kohli's stumps are everywhere!", rec) == set()


def test_well_bowled_praise_is_not_a_wicket_claim():
    assert _violation_slots("Well bowled, just a single conceded.",
                            _record(runs_batter=1, runs_total=1)) == set()


def test_bowled_a_good_over_is_not_a_wicket_claim():
    claims = extract_claims("Khan has bowled a superb over.")
    assert not any(c.slot == "wicket" for c in claims)


def test_hallucinated_wicket():
    assert "wicket" in _violation_slots("He's out! Huge appeal and given!", _record())


def test_wrong_dismissal_type():
    rec = _record(wicket=True, wicket_type="caught", player_out="V Kohli",
                  fielders=["T Boult"])
    assert "dismissal_type" in _violation_slots("Kohli is bowled!", rec)


def test_caught_dismissal_with_fielder_is_clean():
    rec = _record(wicket=True, wicket_type="caught", player_out="V Kohli",
                  fielders=["T Boult"])
    assert _violation_slots("Caught by Boult at long-on! Kohli departs.", rec) == set()


def test_not_out_negation():
    claims = extract_claims("Given not out on review.")
    assert not any(c.slot == "wicket" for c in claims)


def test_wickets_in_hand_state_talk_is_not_a_wicket_claim():
    claims = extract_claims("Six wickets in hand for the chase.")
    assert not any(c.slot == "wicket" for c in claims)


def test_wicketkeeper_is_not_a_wicket_claim():
    claims = extract_claims("Straight to the wicketkeeper, no run.")
    assert not any(c.slot == "wicket" for c in claims)


# --------------------------------------------------------- attribution slot

def test_uninvolved_squad_player_is_misattribution():
    assert "attribution" in _violation_slots("Sharma drives crisply for one.",
                                             _record(runs_batter=1, runs_total=1))


def test_bowler_credit_on_wicket_is_clean_attribution():
    rec = _record(wicket=True, wicket_type="bowled", player_out="V Kohli")
    assert _violation_slots("Khan strikes! Kohli has to go.", rec) == set()


def test_fielder_mention_is_involved():
    rec = _record(wicket=True, wicket_type="run out", player_out="V Kohli",
                  fielders=["J Archer"])
    assert _violation_slots("Run out! Archer with a direct hit, Kohli goes.", rec) == set()


# --------------------------------------------------------------- score slot

def test_score_claim_matches():
    assert _violation_slots("That takes them to 45 for 2.", _record()) == set()
    assert _violation_slots("They move to 45/2.", _record()) == set()


def test_wrong_score_claim():
    assert "score" in _violation_slots("They move to 62 for 3.", _record())


def test_gone_for_personal_score():
    rec = _record(wicket=True, wicket_type="caught", player_out="V Kohli",
                  batter_runs_after=42, fielders=["T Boult"])
    assert _violation_slots("Caught! Kohli departs for 42.", rec) == set()
    assert "personal_score" in _violation_slots("Caught! Kohli departs for 40.", rec)


# ----------------------------------------------------------- milestone slot

def test_fifty_when_crossed_is_clean():
    rec = _record(runs_batter=4, runs_total=4, batter_runs_after=52)
    assert _violation_slots("Four! That brings up his fifty!", rec) == set()


def test_fifty_hallucinated():
    assert "milestone" in _violation_slots("That's his fifty!", _record(batter_runs_after=31))


def test_maiden_fifty_is_a_fifty_not_a_maiden_over():
    rec = _record(runs_batter=1, runs_total=1, batter_runs_after=50)
    assert _violation_slots("A maiden fifty for the young opener!", rec) == set()


def test_maiden_over_checked_against_over_runs():
    clean = _record(completed_over_runs=0)
    assert _violation_slots("A maiden over, superb stuff.", clean) == set()
    leaky = _record(completed_over_runs=7)
    assert "milestone" in _violation_slots("A maiden over, superb stuff.", leaky)


def test_century_hallucinated():
    assert "milestone" in _violation_slots("A magnificent century!", _record(batter_runs_after=88))


# ------------------------------------------------------ omissions & metrics

def test_wicket_omission_detected():
    rec = _record(wicket=True, wicket_type="bowled", player_out="V Kohli")
    res = check_generation("A quiet delivery outside off.", rec)
    assert "wicket unmentioned" in res.omissions


def test_six_omission_detected():
    rec = _record(runs_batter=6, runs_total=6)
    res = check_generation("Runs keep coming for Kohli.", rec)
    assert "six unmentioned" in res.omissions


def test_factcheck_metrics_aggregation():
    rows = [
        {"record": _record(runs_batter=4, runs_total=4)},
        {"record": _record(wicket=True, wicket_type="bowled", player_out="V Kohli")},
        {"record": _record()},
    ]
    gens = [
        "Four! Lovely timing from Kohli.",   # clean
        "A quiet delivery outside off.",     # omission (wicket unmentioned)
        "He's out, surely!",                 # hallucinated wicket
    ]
    metrics = factcheck_metrics(rows, gens, {"spacy_ner": "off"})
    assert metrics["n"] == 3
    # metrics are rounded to 4 decimals before reporting
    assert metrics["hallucination_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["faithfulness_score"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["omission_rate"] == pytest.approx(1 / 2, abs=1e-4)  # of 2 salient rows
    assert metrics["per_slot"]["wicket"]["false_claims"] == 1
    assert metrics["recall"]["wicket_mentioned"] == 0.0
    assert metrics["recall"]["boundary_mentioned"] == 1.0
    assert metrics["ner"]["mode"] == "off"


def test_factcheck_metrics_requires_alignment():
    with pytest.raises(ValueError):
        factcheck_metrics([{"record": _record()}], [], {"spacy_ner": "off"})


# ---------------------------------------------------------------------------
# v2 regression tests: every sentence below produced a WRONG verdict from the
# v1 checker during adversarial review (EXPERIMENTS.md "factcheck-v1-review").
# ---------------------------------------------------------------------------

def test_requirement_talk_is_not_a_runs_claim():
    dot = _record()  # runs_total = 0
    for text in [
        "They need two more to win.",
        "They still need 3 more from this over.",
        "One more needed for victory here.",
        "18 runs needed off the last over.",
    ]:
        assert _violation_slots(text, dot) == set(), text


def test_team_and_partnership_milestones_are_not_personal_claims():
    rec = _record(batter_runs_after=12)
    for text in [
        "They need fifty off the last 30 balls.",
        "Fifty up for the Chargers in the seventh over.",
        "That brings up the fifty partnership!",
        "That's the hundred up for the batting side!",
    ]:
        assert _violation_slots(text, rec) == set(), text


def test_personal_milestone_phrasings_still_checked():
    rec = _record(batter_runs_after=12)
    assert "milestone" in _violation_slots("That brings up Kohli's fifty!", rec)
    assert "milestone" in _violation_slots("Fifty for Kohli, well played.", rec)
    ok = _record(runs_batter=1, runs_total=1, batter_runs_after=50)
    assert _violation_slots("Fifty for Kohli, well played.", ok) == set()


def test_wickets_counts_are_not_boundary_claims():
    dot = _record()
    assert _violation_slots("Six wickets in hand for the chase.", dot) == set()
    assert _violation_slots("Four wickets down and they are wobbling.", dot) == set()


def test_bowler_figures_are_not_score_violations():
    rec = _record()  # team 45/2
    assert _violation_slots("Khan finishes with figures of 2 for 45.", rec) == set()
    assert _violation_slots("Khan ends with 2/45 from his four.", rec) == set()
    # wickets-first scoreboard convention is accepted when it matches
    assert _violation_slots("They are 2 for 45 at the break.", rec) == set()
    # confidently runs-first mismatches still fire
    assert "score" in _violation_slots("They move to 62 for 3.", rec)


def test_absence_of_boundary_phrasings():
    dot = _record()
    assert _violation_slots("Not a single boundary in the last four overs.", dot) == set()
    assert _violation_slots("Ten balls without a boundary now.", dot) == set()


def test_byes_reaching_the_rope_is_a_valid_boundary_mention():
    rec = _record(runs_batter=0, runs_total=4, extras=4, extras_type="byes")
    text = "Four byes as it races past the keeper to the boundary!"
    assert _violation_slots(text, rec) == set()


def test_wicket_idioms_are_not_wicket_claims():
    for text, rec in [
        ("Kohli was caught in two minds there, no run.", _record()),
        ("Mistimed! But it falls safe, and they take a single.",
         _record(runs_batter=1, runs_total=1)),
        ("That's gone, way into the stands! Six!", _record(runs_batter=6, runs_total=6)),
        ("Khan is stumped for answers here, Kohli takes two runs.",
         _record(runs_batter=2, runs_total=2)),
        ("Khan bowled him a vicious bouncer, ducked under it.", _record()),
    ]:
        assert _violation_slots(text, rec) == set(), text


def test_cleaned_him_up_is_a_bowled_claim():
    assert "wicket" in _violation_slots("Cleaned him up! The stumps are shattered!", _record())
    real = _record(wicket=True, wicket_type="bowled", player_out="V Kohli")
    assert _violation_slots("Cleaned him up! The stumps are shattered!", real) == set()


def test_without_doubt_does_not_suppress_a_real_wicket():
    rec = _record(wicket=True, wicket_type="lbw", player_out="V Kohli")
    res = check_generation("Without doubt, that's out! Kohli has to walk.", rec)
    assert any(c.slot == "wicket" for c in res.claims)
    assert res.omissions == []


def test_hyphenated_run_out_checks_dismissal_type():
    rec = _record(wicket=True, wicket_type="caught", player_out="V Kohli",
                  fielders=["T Boult"])
    assert "dismissal_type" in _violation_slots(
        "Run-out! Kohli is short of his ground and gone!", rec)


def test_prospective_maiden_talk_is_unchecked():
    rec = _record(completed_over_runs=None)
    for text in [
        "Khan is two balls away from a maiden here. No run.",
        "A maiden on the cards for Khan.",
    ]:
        assert _violation_slots(text, rec) == set(), text


def test_fielding_and_next_over_mentions_are_not_misattribution():
    dot = _record()
    assert _violation_slots("Driven hard but Sharma at cover cuts it off. No run.", dot) == set()
    assert _violation_slots("Boult will bowl the next over.", dot) == set()


def test_actor_position_still_flags_misattribution():
    rec = _record(runs_batter=1, runs_total=1)
    assert "attribution" in _violation_slots("Sharma sweeps it away for a single.", rec)


def test_plural_boundary_words_count_as_mentions_not_claims():
    six = _record(runs_batter=6, runs_total=6)
    res = check_generation("Back-to-back sixes! Kohli is on fire!", six)
    assert res.omissions == []          # mentioned via plural
    dot = _record()
    assert _violation_slots("Fours and sixes flowed in that over.", dot) == set()


def test_per_slot_precision_books_on_claim_slot():
    rows = [
        {"record": _record(wicket=True, wicket_type="caught", player_out="V Kohli",
                           fielders=["T Boult"])},
        {"record": _record(runs_batter=0, runs_total=4, extras=4, extras_type="legbyes")},
    ]
    gens = [
        "Kohli is bowled!",          # wrong kind -> counts against wicket slot
        "Six leg byes signalled.",   # wrong extras value -> extras slot
    ]
    metrics = factcheck_metrics(rows, gens, {"spacy_ner": "off"})
    wicket = metrics["per_slot"]["wicket"]
    assert wicket["claims"] == 1 and wicket["false_claims"] == 1 and wicket["precision"] == 0.0
    extras = metrics["per_slot"]["extras"]
    assert extras["claims"] == 1 and extras["false_claims"] == 1 and extras["precision"] == 0.0
    for slot, stats in metrics["per_slot"].items():
        if stats["precision"] is not None:
            assert 0.0 <= stats["precision"] <= 1.0, slot
    assert metrics["violations_by_type"]["dismissal_type"] == 1


def test_spacy_auto_degrades_loudly_when_missing():
    metrics = factcheck_metrics(
        [{"record": _record()}], ["No run, defended."],
        {"spacy_ner": "auto", "spacy_model": "en_core_web_sm"},
    )
    ner = metrics["ner"]
    # either the model loaded, or the gap is recorded in the standard
    # unavailable() shape that report.py surfaces under "Metrics not computed"
    assert ner.get("mode") == "spacy" or ner.get("status") == "unavailable"
    if ner.get("status") == "unavailable":
        assert "spacy" in ner["reason"].lower() or "model" in ner["reason"].lower()
