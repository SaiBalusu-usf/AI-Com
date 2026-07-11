"""Game-state features derived per ball (§4): rates, phase, momentum,
event salience, and match tension.

All formulas are the documented v1 (tunable via configs/data.yaml; any change
gets an EXPERIMENTS.md entry). Records must be passed in match order —
momentum and context depend on sequence.
"""

from __future__ import annotations

import math

from .ingest import BallRecord


def _phase(over: int, phases_cfg: dict) -> str:
    for name, (lo, hi) in phases_cfg.items():
        if lo <= over <= hi:
            return name
    # overs beyond the configured ranges (e.g. ODI data through T20 config)
    return "death" if over > max(hi for _, hi in phases_cfg.values()) else "middle"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def current_run_rate(record: BallRecord) -> float:
    if record.balls_bowled == 0:
        return 0.0
    return record.team_score / record.balls_bowled * 6.0


def required_run_rate(record: BallRecord) -> float | None:
    """Runs-per-over still needed in a chase, after this ball. None outside a
    chase or once the innings has no balls left."""
    if record.target is None or record.balls_remaining == 0:
        return None
    needed = record.target - record.team_score
    if needed <= 0:
        return 0.0
    return needed / record.balls_remaining * 6.0


def event_salience(record: BallRecord, phase: str, salience_cfg: dict) -> float:
    if record.wicket:
        return float(salience_cfg["wicket"])
    if record.runs_batter == 6:
        return float(salience_cfg["six"])
    if record.runs_batter == 4:
        return float(salience_cfg["four"])
    if record.runs_total == 0 and phase == "death":
        return float(salience_cfg["dot_in_death"])
    return record.runs_total / 10.0


def tension(
    record: BallRecord,
    phase: str,
    salience: float,
    crr: float,
    rrr: float | None,
    tension_cfg: dict,
) -> float:
    if record.target is not None:  # chase
        cfg = tension_cfg["chase"]
        rrr_val = rrr if rrr is not None else max(crr, 12.0)  # no balls left: max pressure proxy
        pressure = _sigmoid((rrr_val - crr) / float(cfg["rrr_scale"]))
        wickets_in_hand = max(10 - record.team_wickets, 0)
        progress = record.balls_bowled / record.innings_balls_total
        value = (
            float(cfg["rrr_weight"]) * pressure
            + float(cfg["wickets_weight"]) * (1.0 - wickets_in_hand / 10.0)
            + float(cfg["progress_weight"]) * progress
        )
    else:  # first innings
        cfg = tension_cfg["first_innings"]
        phase_weight = float(cfg["phase_weights"].get(phase, 0.5))
        value = float(cfg["phase_weight"]) * phase_weight + float(cfg["salience_weight"]) * salience
    return max(0.0, min(1.0, value))


def compute_features(records: list[BallRecord], features_cfg: dict) -> list[dict]:
    """Feature dict per record. Records must be in match/innings order."""
    window = int(features_cfg["momentum_window_balls"])
    out: list[dict] = []
    # recent (record_key, runs_total) per innings for momentum
    history: dict[tuple[str, int], list[int]] = {}

    for record in records:
        key = (record.match_id, record.innings)
        prev_runs = history.setdefault(key, [])
        momentum = sum(prev_runs[-window:])

        phase = _phase(record.over, features_cfg["phases"])
        crr = current_run_rate(record)
        rrr = required_run_rate(record)
        salience = event_salience(record, phase, features_cfg["salience"])

        out.append(
            {
                "current_run_rate": round(crr, 2),
                "required_run_rate": round(rrr, 2) if rrr is not None else None,
                "wickets_in_hand": max(10 - record.team_wickets, 0),
                "balls_remaining": record.balls_remaining,
                "phase": phase,
                "momentum": momentum,
                "event_salience": round(salience, 3),
                "tension": round(
                    tension(record, phase, salience, crr, rrr, features_cfg["tension"]), 3
                ),
            }
        )
        prev_runs.append(record.runs_total)
    return out
