"""B1 — deterministic rule-based template generator (§6).

Approximates what commercial/EA-style systems ship: slot-filled templates per
outcome type, ~5 surface variants each, chosen by a per-ball seeded RNG so
output is reproducible regardless of iteration order. Zero hallucination by
construction — every slot comes straight from the structured record — so B1
sets the faithfulness ceiling and (with its bounded vocabulary) the diversity
floor.

Game-state conditioning (the RQ2 toggle for this system): when
``use_game_state`` is on, high ``tension x salience`` balls draw from excited
variants; off = always the plain register. Phrasing is intentionally disjoint
from the synthetic reference writer (data/synthetic.py) so surface-overlap
metrics compare real signal, not shared strings.
"""

from __future__ import annotations

from ..data.ingest import BallRecord
from ..utils.seeding import rng_for

_PLAIN = {
    "six": [
        "{batter} hits {bowler} for six.",
        "Six runs, {batter} clears the boundary rope.",
        "{batter} lifts it over the fence for six.",
        "That is a six to {batter}.",
        "{batter} sends it beyond the boundary, six added.",
    ],
    "four": [
        "{batter} hits a four off {bowler}.",
        "Four runs to {batter}.",
        "{batter} pierces the field, four.",
        "A boundary for {batter}.",
        "{batter} guides it to the rope for four.",
    ],
    "dot": [
        "No run off that delivery.",
        "{batter} plays it out, no run.",
        "Dot ball from {bowler}.",
        "{bowler} to {batter}, no run.",
        "Nothing away from that one.",
    ],
    "runs": [
        "{batter} takes {runs_word}.",
        "{runs_word_cap} to {batter}.",
        "{batter} picks up {runs_word} off {bowler}.",
        "They run {runs_word}.",
        "{runs_word_cap} added to the total.",
    ],
}
_EXCITED = {
    "six": [
        "SIX! {batter} goes big off {bowler}!",
        "Enormous from {batter}, that is six!",
        "{batter} deposits {bowler} into the stands, six!",
        "Up and over — six runs, {batter} in charge!",
        "Big hit! Six to {batter}!",
    ],
    "four": [
        "FOUR! {batter} finds the rope!",
        "Terrific stroke — {batter} collects four!",
        "{batter} whips {bowler} away, four!",
        "That flies to the boundary, four to {batter}!",
        "Crisp from {batter}, four runs!",
    ],
    "dot": [
        "Dot ball, and the squeeze is on from {bowler}!",
        "No run — {bowler} holds the nerve!",
        "{batter} can't put it away, dot ball!",
    ],
    "runs": [
        "{batter} hustles for {runs_word}, they need every one!",
        "{runs_word_cap} to {batter}, the chase ticks on!",
    ],
}
_PLAIN_WICKET = {
    "bowled": [
        "{batter} is bowled by {bowler}.",
        "{bowler} hits the stumps, {batter} out bowled.",
        "Bowled him. {batter} goes.",
    ],
    "caught": [
        "{batter} is caught{fielder_by} off {bowler}.",
        "Out. {batter} caught{fielder_by}.",
        "{bowler} takes the wicket, {batter} caught{fielder_by}.",
    ],
    "lbw": [
        "{batter} is lbw to {bowler}.",
        "Out lbw. {batter} departs.",
        "{bowler} traps {batter} lbw.",
    ],
    "run out": [
        "{player_out} is run out{fielder_by}.",
        "Run out. {player_out} is short of the crease.",
    ],
    "stumped": [
        "{batter} is stumped{fielder_by} off {bowler}.",
        "Stumped. {batter} out of his ground.",
    ],
}
_EXCITED_WICKET = {
    "bowled": [
        "BOWLED! {bowler} knocks {batter} over!",
        "The stumps go flying — {batter} bowled by {bowler}!",
    ],
    "caught": [
        "OUT! {batter} caught{fielder_by}, {bowler} strikes!",
        "Taken! {batter} is caught{fielder_by}!",
    ],
    "lbw": [
        "Huge appeal and given — {batter} lbw to {bowler}!",
        "Trapped in front! {batter} is lbw!",
    ],
    "run out": [
        "Direct hit — {player_out} is run out!",
        "Run out! {player_out} is well short!",
    ],
    "stumped": [
        "Lightning work — {batter} stumped{fielder_by}!",
    ],
}
_EXTRAS = {
    "wides": ["Wide ball from {bowler}.", "{bowler} sprays it wide.", "Called wide."],
    "noballs": ["No ball from {bowler}.", "{bowler} oversteps, no ball."],
    "byes": ["They take byes.", "Byes as the keeper misses it."],
    "legbyes": ["Leg byes taken.", "Off the pad, leg byes."],
    "penalty": ["Penalty runs awarded."],
}
_RUNS_WORDS = {1: "a single", 2: "two runs", 3: "three runs", 5: "five runs"}


class TemplateGenerator:
    def __init__(self, cfg: dict, seed: int):
        self.use_game_state = bool(cfg.get("use_game_state", True))
        self.gate = float(cfg.get("excitement_gate", 0.45))
        self.seed = seed

    def generate(self, row: dict) -> str:
        record = row["record"]
        features = row["features"]
        rng = rng_for(
            self.seed, record["match_id"], record["innings"],
            record["over"], record["ball"], "template",
        )
        excited = (
            self.use_game_state
            and features["event_salience"] * features["tension"] > self.gate
        )

        fielder = record["fielders"][0] if record.get("fielders") else ""
        slots = {
            "batter": record["batter"],
            "bowler": record["bowler"],
            "player_out": record["player_out"] or record["batter"],
            "fielder_by": f" by {fielder}" if fielder else "",
            "runs_word": _RUNS_WORDS.get(record["runs_batter"], f"{record['runs_batter']} runs"),
        }
        slots["runs_word_cap"] = slots["runs_word"].capitalize()

        if record["wicket"]:
            kind = record["wicket_type"] if record["wicket_type"] in _PLAIN_WICKET else "caught"
            pool = (_EXCITED_WICKET if excited else _PLAIN_WICKET).get(
                kind, _PLAIN_WICKET[kind]
            )
        elif "noballs" in record["extras_type"] and record["runs_batter"] in (4, 6):
            # a boundary off a no-ball is the ball's headline — the fact
            # checker caught v1 omitting it (EXPERIMENTS.md phase-3-baselines)
            word = "six" if record["runs_batter"] == 6 else "four"
            pool = [
                f"No ball from {{bowler}}, and {{batter}} puts it away for {word}!",
                f"{{bowler}} oversteps and {{batter}} cashes in — {word} runs more!",
            ]
        elif record["extras_type"] != "none":
            base_kind = record["extras_type"].split("+")[0]
            pool = _EXTRAS.get(base_kind, ["Extras signalled."])
        elif record["runs_batter"] == 6:
            pool = (_EXCITED if excited else _PLAIN)["six"]
        elif record["runs_batter"] == 4:
            pool = (_EXCITED if excited else _PLAIN)["four"]
        elif record["runs_total"] == 0:
            pool = _EXCITED["dot"] if excited else _PLAIN["dot"]
        else:
            pool = _EXCITED["runs"] if excited else _PLAIN["runs"]

        line = str(rng.choice(pool)).format(**slots)

        crossed_50 = (
            record["batter_runs_after"] >= 50
            and record["batter_runs_after"] - record["runs_batter"] < 50
        )
        crossed_100 = (
            record["batter_runs_after"] >= 100
            and record["batter_runs_after"] - record["runs_batter"] < 100
        )
        if crossed_100:
            line += f" {record['batter']} reaches his century."
        elif crossed_50:
            line += f" Fifty for {record['batter']}."
        if rng.random() < 0.2:
            line += f" Score {record['team_score']}/{record['team_wickets']}."
        return line


def build(cfg: dict, seed: int, train_rows: list[dict] | None = None) -> TemplateGenerator:
    return TemplateGenerator(cfg, seed)
