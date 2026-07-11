"""Seeded synthetic T20 simulator emitting EXACT Cricsheet-format match JSON,
plus a reference-commentary writer.

Purpose (PLAN.md, "Session environment constraints"): cricsheet.org and every
parallel-commentary source are unreachable from the build environment, but the
pipeline, fixture, tests and golden path must run end-to-end. The simulator
produces matches in the same schema `ingest.parse_match` consumes for real
Cricsheet data, so swapping in real data changes nothing downstream. Every row
built from this module is tagged ``data_source=synthetic`` and
``reference_source=synthetic_v1`` — it is pipeline-validation data, never
paper numbers.

The reference writer is intentionally DIFFERENT from the B1 template baseline
(different vocabulary, structures, and noise), and its excitement level
correlates with ``event_salience x tension`` by construction with noise — so
the §5.2 excitement-scorer validation measures something real at fixture
scale.

Everything is driven by numpy Generators derived from
``utils.seeding.rng_for`` — same seed, same matches, same references.
"""

from __future__ import annotations

import numpy as np

from ..utils.seeding import rng_for
from .ingest import BallRecord

# ---------------------------------------------------------------- name pools
# All names are fictional (no real players/teams implied).
_TEAMS = [
    "Harborview Herons", "Ironpeak Falcons", "Monsoon Riders", "Saffron Kings",
    "Basalt Bay Blasters", "Velvet Valley Vipers", "Northgate Nomads", "Coral City Chargers",
]
_SURNAMES = [
    "Advani", "Barretto", "Chandran", "Deshpande", "Ellappan", "Farooqi",
    "Gaikwad", "Hazarika", "Irani", "Jadhav", "Kulkarni", "Lakhani",
    "Mistry", "Naidu", "Oberoi", "Pillai", "Qureshi", "Rautela",
    "Sarwate", "Thampi", "Uthappa", "Vaswani", "Wadekar", "Yandamuri",
    "Zutshi", "Ambrose", "Blackwood", "Cummins", "Delacroix", "Everton",
    "Fitzroy", "Greaves", "Holloway", "Ives", "Jennings", "Kirkland",
    "Lambert", "Mortlock", "Nesbitt", "Ockwell", "Pemberton", "Quill",
    "Rothwell", "Silvers", "Tremblay", "Underhill", "Verlander", "Whitfield",
]
_INITIALS = list("ABCDEFGHJKLMNPRSTV")

_WICKET_KINDS = ["caught", "bowled", "lbw", "run out", "stumped"]
_WICKET_P = [0.58, 0.20, 0.10, 0.07, 0.05]

# per-phase outcome distributions: (batter_runs or event, probability)
_OUTCOMES = ["0", "1", "2", "3", "4", "6", "W", "wd", "nb", "b", "lb"]
_PHASE_P = {
    "powerplay": [0.31, 0.29, 0.065, 0.008, 0.14, 0.055, 0.045, 0.035, 0.010, 0.006, 0.036],
    "middle":    [0.28, 0.36, 0.090, 0.010, 0.09, 0.040, 0.050, 0.035, 0.008, 0.005, 0.032],
    "death":     [0.212, 0.30, 0.080, 0.012, 0.15, 0.110, 0.075, 0.035, 0.012, 0.004, 0.010],
}
# guard against typos, then normalize away float drift so rng.choice accepts
for _phase_name, _p in _PHASE_P.items():
    if abs(sum(_p) - 1.0) > 0.01:
        raise ValueError(f"outcome probabilities for {_phase_name} sum to {sum(_p)}")
    _total = sum(_p)
    _PHASE_P[_phase_name] = [x / _total for x in _p]


def _phase_of_over(over_1based: int) -> str:
    if over_1based <= 6:
        return "powerplay"
    if over_1based <= 15:
        return "middle"
    return "death"


def _make_squads(rng: np.random.Generator) -> tuple[list[str], list[str], list[str], list[str]]:
    team_a, team_b = (str(t) for t in rng.choice(_TEAMS, size=2, replace=False))
    surnames = [str(s) for s in rng.choice(_SURNAMES, size=22, replace=False)]
    players = [f"{rng.choice(_INITIALS)} {s}" for s in surnames]
    return team_a, team_b, players[:11], players[11:]


def _simulate_innings(
    rng: np.random.Generator,
    batting: list[str],
    bowling: list[str],
    overs_total: int,
    target: int | None,
) -> list[dict]:
    """Return Cricsheet-format over blocks for one innings."""
    striker, non_striker = batting[0], batting[1]
    next_batter = 2
    score, wickets = 0, 0
    bowler_pool = bowling[5:11]  # last six players bowl
    last_bowler = None
    over_blocks: list[dict] = []

    for over_idx in range(overs_total):
        choices = [b for b in bowler_pool if b != last_bowler]
        bowler = str(rng.choice(choices))
        last_bowler = bowler
        deliveries: list[dict] = []
        legal = 0
        while legal < 6:
            phase = _phase_of_over(over_idx + 1)
            outcome = str(rng.choice(_OUTCOMES, p=_PHASE_P[phase]))
            delivery: dict = {
                "batter": striker,
                "bowler": bowler,
                "non_striker": non_striker,
                "runs": {"batter": 0, "extras": 0, "total": 0},
            }
            batter_runs, extras, extras_field = 0, 0, None
            wicket = None

            if outcome == "W":
                kind = str(rng.choice(_WICKET_KINDS, p=_WICKET_P))
                wicket = {"kind": kind, "player_out": striker}
                if kind in ("caught", "stumped", "run out"):
                    wicket["fielders"] = [{"name": str(rng.choice(bowling[:11]))}]
            elif outcome == "wd":
                extras = int(rng.choice([1, 1, 1, 2, 5], p=[0.82, 0.05, 0.05, 0.05, 0.03]))
                extras_field = {"wides": extras}
            elif outcome == "nb":
                bat_off_nb = int(rng.choice([0, 1, 2, 4, 6], p=[0.45, 0.30, 0.10, 0.10, 0.05]))
                batter_runs, extras = bat_off_nb, 1
                extras_field = {"noballs": 1}
            elif outcome == "b":
                extras = int(rng.choice([1, 2, 4], p=[0.7, 0.2, 0.1]))
                extras_field = {"byes": extras}
            elif outcome == "lb":
                extras = int(rng.choice([1, 2], p=[0.85, 0.15]))
                extras_field = {"legbyes": extras}
            else:
                batter_runs = int(outcome)

            total = batter_runs + extras
            delivery["runs"] = {"batter": batter_runs, "extras": extras, "total": total}
            if extras_field:
                delivery["extras"] = extras_field
            if wicket:
                delivery["wickets"] = [wicket]

            deliveries.append(delivery)
            score += total
            is_legal = outcome not in ("wd", "nb")
            if is_legal:
                legal += 1

            if wicket:
                wickets += 1
                if wickets >= 10 or next_batter >= 11:
                    return over_blocks + [{"over": over_idx, "deliveries": deliveries}]
                striker = batting[next_batter]
                next_batter += 1
            else:
                # strike rotation on odd TOTAL runs for byes/legbyes/wides,
                # odd batter runs otherwise
                rotating = total if extras_field and "wides" in (extras_field or {}) else (
                    total if extras_field else batter_runs
                )
                if rotating % 2 == 1:
                    striker, non_striker = non_striker, striker

            if target is not None and score >= target:
                return over_blocks + [{"over": over_idx, "deliveries": deliveries}]

        over_blocks.append({"over": over_idx, "deliveries": deliveries})
        striker, non_striker = non_striker, striker  # over-end swap

    return over_blocks


def simulate_match(seed: int, match_no: int, overs: int = 20) -> tuple[str, dict]:
    """One full Cricsheet-format T20 match dict, deterministically from (seed, match_no)."""
    rng = rng_for(seed, "match", match_no)
    team_a, team_b, squad_a, squad_b = _make_squads(rng)
    match_id = f"synth_{seed}_{match_no:04d}"

    first = _simulate_innings(rng, squad_a, squad_b, overs, target=None)
    first_total = sum(d["runs"]["total"] for o in first for d in o["deliveries"])
    target = first_total + 1
    second = _simulate_innings(rng, squad_b, squad_a, overs, target=target)

    match = {
        "meta": {"data_version": "1.1.0", "created": "2026-01-01", "revision": 1},
        "info": {
            "match_type": "T20",
            "overs": overs,
            "teams": [team_a, team_b],
            "players": {team_a: squad_a, team_b: squad_b},
            "dates": ["2026-01-01"],
            "venue": "Synthetic Oval",
            "event": {"name": "Synthetic Premier League"},
            "synthetic": True,
        },
        "innings": [
            {"team": team_a, "overs": first},
            {"team": team_b, "overs": second, "target": {"runs": target, "overs": overs}},
        ],
    }
    return match_id, match


def simulate_matches(seed: int, n_matches: int, overs: int = 20) -> dict[str, dict]:
    return dict(simulate_match(seed, i, overs) for i in range(n_matches))


# ------------------------------------------------------- reference commentary
# Vocabulary deliberately disjoint from models/template.py where possible.

_EXCITED_WICKET = [
    "GONE! {bowler} strikes, {batter} has to go — {kind_phrase}!",
    "That's a huge breakthrough! {batter} departs, {kind_phrase}, and {bowler} roars!",
    "OUT! {kind_phrase} and {batter}'s brilliant stay is over!",
    "What a moment! {bowler} gets his man, {batter} {kind_phrase_short}!",
]
_PLAIN_WICKET = [
    "{batter} is out, {kind_phrase}. {bowler} gets the wicket.",
    "End of the innings for {batter}, {kind_phrase}.",
    "{bowler} removes {batter}, {kind_phrase}.",
]
_EXCITED_SIX = [
    "SIX! {batter} launches {bowler} miles over the rope!",
    "That is absolutely massive from {batter}! Into the crowd!",
    "Huge hit! {batter} climbs into that one and it sails away for six!",
    "{batter} unloads — that's out of here! Maximum!",
]
_PLAIN_SIX = [
    "{batter} clears the boundary off {bowler}, six runs.",
    "Six for {batter}, comfortably over the rope.",
]
_EXCITED_FOUR = [
    "Cracking shot! {batter} threads it away and it races to the fence!",
    "FOUR! {batter} times that beautifully off {bowler}!",
    "Glorious from {batter} — the fielder had no chance, four runs!",
]
_PLAIN_FOUR = [
    "{batter} finds the gap off {bowler}, four runs added.",
    "Nicely placed by {batter}, away for four.",
    "Four to {batter}, beating the infield.",
]
_PLAIN_DOT = [
    "No run, {batter} defends solidly.",
    "Dot ball. {bowler} keeps it tight.",
    "{batter} lets that one go through to the keeper.",
    "Pushed straight to the fielder by {batter}, no run.",
]
_EXCITED_DOT = [
    "Superb from {bowler}! {batter} can't get it away — the pressure builds!",
    "Another dot! {bowler} is squeezing hard here!",
]
_PLAIN_RUNS = [
    "{batter} works it away for {runs_word}.",
    "{runs_word_cap} to {batter}, tidy cricket.",
    "{batter} nudges {bowler} into the gap, {runs_word}.",
    "Quick running from {batter}, {runs_word} taken.",
]
_EXCITED_RUNS = [
    "{batter} scampers back — {runs_word} more, every run gold here!",
    "Smart cricket! {batter} keeps the board ticking with {runs_word}!",
]
_PLAIN_EXTRA = {
    "wides": ["{bowler} strays down the side, wide called.", "Wide from {bowler}, a gift."],
    "noballs": ["No ball! {bowler} oversteps.", "{bowler} overshoots the crease, no ball signalled."],
    "byes": ["Through everyone, byes taken.", "The keeper can't gather, byes."],
    "legbyes": ["Off the pads, leg byes.", "Deflected away, they cross for leg byes."],
}
_RUNS_WORDS = {1: "a single", 2: "two", 3: "three", 5: "five"}


def _kind_phrase(record: BallRecord) -> tuple[str, str]:
    kind = record.wicket_type
    fielder = record.fielders[0] if record.fielders else ""
    if kind == "caught":
        return (f"caught by {fielder}" if fielder else "taken in the deep", "is caught")
    if kind == "bowled":
        return ("the stumps are shattered", "is bowled")
    if kind == "lbw":
        return ("trapped plumb in front", "is lbw")
    if kind == "run out":
        return (f"run out by {fielder}" if fielder else "run out by a direct hit", "is run out")
    if kind == "stumped":
        return (f"stumped by {fielder}" if fielder else "stumped off a wide one", "is stumped")
    return (kind or "dismissed", "is out")


def write_reference(record: BallRecord, features: dict, seed: int) -> str:
    """Human-style reference line whose excitement tracks salience x tension."""
    rng = rng_for(seed, record.match_id, record.innings, record.over, record.ball, "ref")
    heat = features["event_salience"] * features["tension"]
    excited = (heat * 1.7 + rng.normal(0, 0.18)) > 0.45

    kind_phrase, kind_phrase_short = ("", "")
    if record.wicket:
        kind_phrase, kind_phrase_short = _kind_phrase(record)

    slots = {
        "batter": record.batter,
        "bowler": record.bowler,
        "kind_phrase": kind_phrase,
        "kind_phrase_short": kind_phrase_short,
        "runs_word": _RUNS_WORDS.get(record.runs_batter, f"{record.runs_batter} runs"),
    }
    slots["runs_word_cap"] = slots["runs_word"].capitalize()

    if record.wicket:
        pool = _EXCITED_WICKET if excited else _PLAIN_WICKET
    elif record.runs_batter == 6:
        pool = _EXCITED_SIX if excited else _PLAIN_SIX
    elif record.runs_batter == 4:
        pool = _EXCITED_FOUR if excited else _PLAIN_FOUR
    elif record.extras_type != "none":
        base = record.extras_type.split("+")[0]
        pool = _PLAIN_EXTRA.get(base, ["Extras signalled."])
    elif record.runs_total == 0:
        pool = _EXCITED_DOT if excited else _PLAIN_DOT
    else:
        pool = _EXCITED_RUNS if excited else _PLAIN_RUNS

    line = str(rng.choice(pool)).format(**slots)

    # occasional post-ball score mention (kept factual by construction)
    if rng.random() < 0.22:
        line += f" {record.batting_team} are {record.team_score} for {record.team_wickets}."
    # milestone sentences when the batter just crossed the mark
    crossed_50 = record.batter_runs_after >= 50 and record.batter_runs_after - record.runs_batter < 50
    crossed_100 = record.batter_runs_after >= 100 and record.batter_runs_after - record.runs_batter < 100
    if crossed_100:
        line += f" That's a magnificent century for {record.batter}!"
    elif crossed_50:
        line += f" That brings up {record.batter}'s fifty!"
    # chase pressure flavour in the death overs
    if (
        record.target is not None
        and features["phase"] == "death"
        and not record.wicket
        and rng.random() < 0.3
    ):
        need = record.target - record.team_score
        if need > 0:
            line += f" {need} needed off {record.balls_remaining}."
    return line
