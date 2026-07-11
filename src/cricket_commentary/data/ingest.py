"""Parse Cricsheet-format match JSON into canonical per-ball records.

One parser for both real Cricsheet downloads and the synthetic simulator
(data/synthetic.py), which emits the identical schema — so the whole pipeline
downstream of this file is source-agnostic.

Conventions (documented once, relied on everywhere):

- ``innings`` is 1-based; super-over innings are skipped.
- ``over`` is 1-based (over 1..20 for T20) so phase configs read naturally.
  Cricket scoreboard notation ("17.3" = 3rd legal ball of the 18th over) is
  produced by :func:`over_ball_display`.
- ``ball`` is the legal-ball number within the over (1..6). Wides/no-balls do
  not advance it: they carry the number of the upcoming legal delivery and
  ``is_legal`` False.
- ``team_score`` / ``team_wickets`` / ``batter_runs_after`` are cumulative
  INCLUDING the current delivery — commentary like "that takes them to 162/4"
  refers to the post-ball score.
- ``balls_remaining`` counts legal deliveries left in the innings after this
  one (never negative; innings cut short by a chase still compute vs. the
  scheduled overs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Cricsheet wicket kinds that do not cost the batting side a wicket.
_NOT_A_DISMISSAL = {"retired hurt", "retired not out"}


@dataclass
class BallRecord:
    match_id: str
    innings: int
    over: int                      # 1-based
    ball: int                      # legal-ball number within over, 1-based
    is_legal: bool
    batting_team: str
    bowling_team: str
    batter: str
    bowler: str
    non_striker: str
    runs_batter: int
    runs_total: int                # batter runs + extras off this delivery
    extras: int
    extras_type: str               # "wides", "noballs", "byes", "legbyes", "penalty", combos "wides+penalty", or "none"
    wicket: bool
    wicket_type: str               # "" when no wicket
    player_out: str                # "" when no wicket
    fielders: list[str] = field(default_factory=list)
    team_score: int = 0            # after this ball
    team_wickets: int = 0          # after this ball
    target: int | None = None      # runs to win (2nd innings), else None
    balls_remaining: int = 0       # legal balls left after this one
    balls_bowled: int = 0          # legal balls bowled including this one (0 if illegal? no: cumulative count)
    innings_balls_total: int = 120
    batter_runs_after: int = 0     # batter's personal score after this ball
    batter_balls_faced: int = 0    # legal balls faced by batter incl. this one
    completed_over_runs: int | None = None  # total runs in the over, set on its final ball (maiden check)
    squad: list[str] = field(default_factory=list)  # all player names in the match (misattribution checks)

    def to_dict(self) -> dict:
        return asdict(self)


def over_ball_display(record: BallRecord | dict) -> str:
    over = record.over if isinstance(record, BallRecord) else record["over"]
    ball = record.ball if isinstance(record, BallRecord) else record["ball"]
    return f"{over - 1}.{ball}"


def _extras_type(delivery: dict) -> str:
    kinds = sorted((delivery.get("extras") or {}).keys())
    return "+".join(kinds) if kinds else "none"


def _is_legal(delivery: dict) -> bool:
    extras = delivery.get("extras") or {}
    return "wides" not in extras and "noballs" not in extras


def parse_match(match: dict, match_id: str) -> list[BallRecord]:
    """Flatten one Cricsheet-format match dict into ordered BallRecords."""
    info = match.get("info") or {}
    teams = list(info.get("teams") or [])
    if len(teams) != 2:
        raise ValueError(f"{match_id}: expected exactly 2 teams, got {teams!r}")
    overs_scheduled = int(info.get("overs") or 20)
    squad = sorted({p for names in (info.get("players") or {}).values() for p in names})

    records: list[BallRecord] = []
    innings_no = 0
    for innings in match.get("innings") or []:
        if innings.get("super_over"):
            continue
        innings_no += 1
        batting_team = innings["team"]
        bowling_team = teams[0] if batting_team == teams[1] else teams[1]
        target = None
        if isinstance(innings.get("target"), dict):
            target = int(innings["target"].get("runs"))

        innings_balls_total = overs_scheduled * 6
        score = 0
        wickets = 0
        legal_bowled = 0
        batter_runs: dict[str, int] = {}
        batter_balls: dict[str, int] = {}

        for over_block in innings.get("overs") or []:
            over_1based = int(over_block["over"]) + 1
            legal_in_over = 0
            over_runs = 0
            deliveries = over_block.get("deliveries") or []
            # find the index of the last legal delivery to stamp completed_over_runs
            last_legal_idx = max(
                (i for i, d in enumerate(deliveries) if _is_legal(d)), default=None
            )
            for idx, delivery in enumerate(deliveries):
                runs = delivery.get("runs") or {}
                runs_batter = int(runs.get("batter", 0))
                runs_extras = int(runs.get("extras", 0))
                runs_total = int(runs.get("total", runs_batter + runs_extras))
                legal = _is_legal(delivery)
                if legal:
                    legal_bowled += 1
                    legal_in_over += 1
                over_runs += runs_total
                score += runs_total

                wicket_events = [
                    w for w in (delivery.get("wickets") or [])
                    if w.get("kind") not in _NOT_A_DISMISSAL
                ]
                wickets += len(wicket_events)
                first_wicket = wicket_events[0] if wicket_events else {}
                fielders = [
                    f.get("name", "") if isinstance(f, dict) else str(f)
                    for f in (first_wicket.get("fielders") or [])
                ]

                batter = delivery["batter"]
                batter_runs[batter] = batter_runs.get(batter, 0) + runs_batter
                if legal:
                    batter_balls[batter] = batter_balls.get(batter, 0) + 1

                is_final_legal_of_over = (
                    idx == last_legal_idx and legal_in_over == 6
                )
                records.append(
                    BallRecord(
                        match_id=match_id,
                        innings=innings_no,
                        over=over_1based,
                        ball=min(legal_in_over + (0 if legal else 1), 6),
                        is_legal=legal,
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        batter=batter,
                        bowler=delivery["bowler"],
                        non_striker=delivery.get("non_striker", ""),
                        runs_batter=runs_batter,
                        runs_total=runs_total,
                        extras=runs_extras,
                        extras_type=_extras_type(delivery),
                        wicket=bool(wicket_events),
                        wicket_type=first_wicket.get("kind", ""),
                        player_out=first_wicket.get("player_out", ""),
                        fielders=fielders,
                        team_score=score,
                        team_wickets=wickets,
                        target=target,
                        balls_remaining=max(innings_balls_total - legal_bowled, 0),
                        balls_bowled=legal_bowled,
                        innings_balls_total=innings_balls_total,
                        batter_runs_after=batter_runs[batter],
                        batter_balls_faced=batter_balls.get(batter, 0),
                        completed_over_runs=over_runs if is_final_legal_of_over else None,
                        squad=squad,
                    )
                )
    return records


def parse_matches(matches: dict[str, dict]) -> list[BallRecord]:
    """Parse {match_id: match_dict}, keeping records grouped and ordered by match."""
    out: list[BallRecord] = []
    for match_id in sorted(matches):
        out.extend(parse_match(matches[match_id], match_id))
    return out
