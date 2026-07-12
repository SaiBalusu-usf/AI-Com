"""Linearize a (record, features) pair into the model input string (§4).

Two formats, ablatable via config:
- flat:    ``over=17.3 | batter=Kohli | bowler=Rashid Khan | outcome=SIX |
           runs=6 | score=162/4 | RRR=9.8 | wkts_in_hand=6 | phase=death``
- context: same, prefixed with compressed outcomes of the previous N balls:
           ``prev=FOUR,DOT | over=...``

Game-state fields (RRR, wkts_in_hand, phase, momentum) are emitted only when
``include_game_state`` is true — this toggle IS the RQ2 experiment.
"""

from __future__ import annotations

from .ingest import BallRecord, over_ball_display

_RUN_WORDS = {0: "DOT", 1: "SINGLE", 2: "TWO", 3: "THREE", 5: "FIVE"}


def outcome_label(record: BallRecord) -> str:
    """Compressed outcome, most newsworthy fact first."""
    if record.wicket:
        kind = (record.wicket_type or "wicket").upper().replace(" ", "_")
        return f"WICKET({kind})"
    if record.runs_batter == 6:
        return "SIX"
    if record.runs_batter == 4:
        return "FOUR"
    if record.extras_type != "none":
        tag = record.extras_type.upper().replace("+", "_")
        return f"{tag}{record.extras}" if record.extras else tag
    return _RUN_WORDS.get(record.runs_batter, f"RUNS({record.runs_batter})")


def linearize(
    record: BallRecord,
    features: dict,
    fmt: str = "flat",
    include_game_state: bool = True,
    prev_outcomes: list[str] | None = None,
    context_balls: int = 2,
) -> str:
    """``prev_outcomes`` are outcome labels of the PRECEDING DELIVERIES of the
    innings in order (computed at dataset build time, before any cleaning —
    reconstructing them from surviving rows would silently skip dropped
    deliveries and prefix factually wrong context)."""
    parts = [
        f"over={over_ball_display(record)}",
        f"batter={record.batter}",
        f"bowler={record.bowler}",
        f"outcome={outcome_label(record)}",
        f"runs={record.runs_total}",
        f"score={record.team_score}/{record.team_wickets}",
    ]
    if record.wicket:
        parts.insert(4, f"player_out={record.player_out}")
    if include_game_state:
        rrr = features.get("required_run_rate")
        if rrr is not None:
            parts.append(f"RRR={rrr}")
        if record.target is not None:
            parts.append(f"target={record.target}")
        parts.append(f"wkts_in_hand={features['wickets_in_hand']}")
        parts.append(f"phase={features['phase']}")
        parts.append(f"momentum={features['momentum']}")

    line = " | ".join(parts)
    if fmt == "context":
        prev = (prev_outcomes or [])[-context_balls:]
        prefix = f"prev={','.join(prev) if prev else 'NONE'} | "
        line = prefix + line
    elif fmt != "flat":
        raise ValueError(f"unknown linearization format: {fmt}")
    return line
