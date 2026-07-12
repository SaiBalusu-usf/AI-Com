"""D1 loader: parallel ball-by-ball commentary CSVs (Kaggle) → the pipeline.

Design: rather than joining Kaggle commentary onto separately-downloaded
Cricsheet matches (which needs fragile cross-source player/match matching),
the CSV's own structured columns are converted into Cricsheet-format match
dicts and fed through the SAME ``ingest.parse_match`` as every other source.
Batter/bowler names then share the commentary's own spelling — exactly what
the fact checker's attribution slot needs — and cumulative state (scores,
balls remaining, batter tallies) is derived by the audited ingest math
instead of being trusted from the CSV.

Column names differ per dataset, so the mapping lives in configs/data.yaml
(``source.kaggle_commentary.columns``) with defaults covering the common
conventions of the two candidate datasets (PLAN.md D1):

- ``balabaskar/asia-cup-2022-ball-by-ball-data-and-commentary``
- ``amitkumarbhowmick/ipl-2024-ball-by-ball-commentary-dataset``

Neither dataset's declared licence is verifiable from the build environment
(kaggle.com is egress-blocked); the licence shown on the Kaggle page MUST be
recorded in data/DATASET_CARD.md at download time, and both are presumed
scraped from a commentary site — a provenance caveat the card carries
regardless of the uploader's licence choice.

A mandatory alignment audit fails loudly when the configured column map is
likely wrong (§2 rule 7): too many rows dropping as entity mismatches means
names and text are not from the same delivery.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("kaggle_commentary")

# accepted per-logical-field header names, tried in order, case-insensitive
DEFAULT_COLUMNS: dict[str, list[str]] = {
    "match_id": ["match_id", "matchid", "match", "match_no", "match_name"],
    "innings": ["innings", "inning", "innings_no"],
    "over_ball": ["over_ball", "over.ball", "ball_no"],   # combined "17.3"
    "over": ["over", "over_no", "overs"],                 # separate ints
    "ball": ["ball", "ball_in_over", "delivery"],
    "batting_team": ["batting_team", "battingteam", "team_batting", "team"],
    "batter": ["batter", "batsman", "striker", "batsman_name"],
    "bowler": ["bowler", "bowler_name"],
    "non_striker": ["non_striker", "nonstriker", "non_striker_name"],
    "runs_batter": ["runs_batter", "batsman_runs", "runs_off_bat", "batter_runs"],
    "runs_total": ["runs_total", "total_runs", "runs", "total_run"],
    "extras": ["extras", "extra_runs", "extras_run"],
    "extras_type": ["extras_type", "extra_type", "kind_of_extras"],
    "wicket": ["wicket", "is_wicket", "iswicket", "wicket_fallen"],
    "wicket_type": ["wicket_type", "dismissal_kind", "kind", "dismissal"],
    "player_out": ["player_out", "player_dismissed", "dismissed_player"],
    "commentary": ["commentary", "commentary_text", "text", "comment", "description"],
}

_WIDE_MARKERS = ("wide", "wd")
_NOBALL_MARKERS = ("noball", "no ball", "no-ball", "nb")


class ColumnMapError(Exception):
    pass


def resolve_columns(header: list[str], overrides: dict | None = None) -> dict[str, str]:
    """Map logical fields -> actual CSV headers. Explicit overrides from the
    config win; otherwise the first DEFAULT_COLUMNS candidate present is used.
    Fails loudly when required fields cannot be located."""
    lower = {h.lower().strip(): h for h in header}
    resolved: dict[str, str] = {}
    for field, candidates in DEFAULT_COLUMNS.items():
        override = (overrides or {}).get(field)
        if override:
            if override.lower() not in lower:
                raise ColumnMapError(
                    f"configured column {override!r} for field {field!r} not in "
                    f"CSV header {header}"
                )
            resolved[field] = lower[override.lower()]
            continue
        for cand in candidates:
            if cand in lower:
                resolved[field] = lower[cand]
                break

    required = ["batter", "bowler", "commentary"]
    missing = [f for f in required if f not in resolved]
    if "over_ball" not in resolved and "over" not in resolved:
        missing.append("over (or combined over_ball)")
    if missing:
        raise ColumnMapError(
            f"cannot locate required fields {missing} in CSV header {header}; "
            "set source.kaggle_commentary.columns in configs/data.yaml"
        )
    return resolved


def _parse_over_ball(row: dict, cols: dict[str, str]) -> tuple[int, int]:
    """Return (over_0based, ball) from combined '17.3' or separate columns.
    Combined notation is scoreboard-style: 0-based over, 1-based ball."""
    if "over_ball" in cols:
        raw = str(row[cols["over_ball"]]).strip()
        over_part, _, ball_part = raw.partition(".")
        return int(float(over_part)), int(ball_part or 1)
    over = int(float(row[cols["over"]]))
    ball = int(float(row.get(cols.get("ball", ""), 1) or 1))
    # some datasets store 1-based overs; normalise to 0-based cricsheet blocks
    return max(over - 1, 0) if _overs_look_one_based(row, cols) else over, ball


def _overs_look_one_based(row: dict, cols: dict[str, str]) -> bool:
    # decided once per file by the caller and stashed on the row dict
    return bool(row.get("_overs_one_based"))


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def load_commentary_csv(
    path: str | Path, column_overrides: dict | None = None
) -> tuple[dict[str, dict], dict[tuple, str]]:
    """Parse one commentary CSV into ({match_id: cricsheet_dict},
    {(match_id, innings, over_1based, delivery_seq): commentary_text})."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ColumnMapError(f"{path}: empty CSV")
        cols = resolve_columns(list(reader.fieldnames), column_overrides)
        rows = list(reader)
    if not rows:
        raise ColumnMapError(f"{path}: no data rows")

    # over-numbering convention: explicit config override wins; otherwise a
    # per-file heuristic (any over 0 => 0-based). A 0-based file that happens
    # to lack over-0 rows would be misread — set the override in that case.
    if "over" in cols and "over_ball" not in cols:
        override = (column_overrides or {}).get("overs_one_based")
        if override is not None:
            one_based = bool(override)
        else:
            one_based = min(
                int(float(r[cols["over"]])) for r in rows if r[cols["over"]]
            ) >= 1
        for r in rows:
            r["_overs_one_based"] = one_based
    stats: Counter = Counter(rows=len(rows))

    # group deliveries per (match, innings), preserving file order
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        match_id = str(row.get(cols.get("match_id", ""), "match_0") or "match_0").strip()
        innings = int(float(row.get(cols.get("innings", ""), 1) or 1))
        grouped.setdefault((match_id, innings), []).append(row)

    matches: dict[str, dict] = {}
    reference_texts: dict[tuple, str] = {}

    for (match_id, innings_no), deliveries in sorted(grouped.items()):
        match = matches.setdefault(
            match_id,
            {"info": {"match_type": "T20", "overs": 20, "teams": [], "players": {}},
             "innings": []},
        )
        overs_blocks: dict[int, list[dict]] = {}
        batting_team = ""
        players: set[str] = set()
        seq_per_over: Counter = Counter()

        for row in deliveries:
            over0, _ball = _parse_over_ball(row, cols)
            batter = str(row[cols["batter"]]).strip()
            bowler = str(row[cols["bowler"]]).strip()
            non_striker = str(row.get(cols.get("non_striker", ""), "") or "").strip()
            players.update(p for p in (batter, bowler, non_striker) if p)
            batting_team = batting_team or str(
                row.get(cols.get("batting_team", ""), "") or f"{match_id}-batting-{innings_no}"
            ).strip()

            extras = int(float(row.get(cols.get("extras", ""), 0) or 0))
            runs_batter_raw = row.get(cols.get("runs_batter", ""), None)
            runs_total_raw = row.get(cols.get("runs_total", ""), None)
            if runs_batter_raw is None and runs_total_raw is None:
                raise ColumnMapError(
                    f"{path}: neither batter-runs nor total-runs column found"
                )
            runs_batter = int(float(runs_batter_raw or 0)) if runs_batter_raw is not None else max(
                int(float(runs_total_raw or 0)) - extras, 0
            )
            total = int(float(runs_total_raw)) if runs_total_raw not in (None, "") else runs_batter + extras

            delivery: dict = {
                "batter": batter,
                "bowler": bowler,
                "non_striker": non_striker,
                "runs": {"batter": runs_batter, "extras": extras, "total": total},
            }

            extras_kind = str(row.get(cols.get("extras_type", ""), "") or "").strip().lower()
            if extras > 0:
                if any(m in extras_kind for m in _WIDE_MARKERS):
                    delivery["extras"] = {"wides": extras}
                elif any(m in extras_kind for m in _NOBALL_MARKERS):
                    delivery["extras"] = {"noballs": extras}
                elif "bye" in extras_kind and "leg" in extras_kind:
                    delivery["extras"] = {"legbyes": extras}
                elif "bye" in extras_kind:
                    delivery["extras"] = {"byes": extras}
                elif "penalty" in extras_kind:
                    delivery["extras"] = {"penalty": extras}
                else:
                    # unknown typing: counted as byes so runs stay right, but
                    # the delivery is treated as LEGAL — flagged in the audit
                    # because wides/no-balls misclassified here shift
                    # balls_remaining
                    delivery["extras"] = {"byes": extras}
                    stats["extras_type_unknown"] += 1

            wicket_kind = str(row.get(cols.get("wicket_type", ""), "") or "").strip().lower()
            player_out = str(row.get(cols.get("player_out", ""), "") or "").strip()
            wicket_flag = _truthy(row.get(cols.get("wicket", ""), "")) or bool(
                wicket_kind and wicket_kind not in ("nan", "none", "not out")
            )
            if wicket_flag:
                delivery["wickets"] = [{
                    "kind": wicket_kind or "unknown",
                    "player_out": player_out or batter,
                }]

            overs_blocks.setdefault(over0, []).append(delivery)
            seq_per_over[over0] += 1
            text = str(row[cols["commentary"]] or "").strip()
            if text:
                reference_texts[
                    (match_id, innings_no, over0 + 1, seq_per_over[over0])
                ] = text
            else:
                stats["empty_commentary"] += 1

        match["innings"].append(
            {"team": batting_team,
             "overs": [{"over": o, "deliveries": ds} for o, ds in sorted(overs_blocks.items())]}
        )
        info = match["info"]
        info["players"][batting_team] = sorted(players)
        if batting_team not in info["teams"]:
            info["teams"].append(batting_team)

    # two-team invariant expected by ingest; single-innings files get a stub
    for match in matches.values():
        teams = match["info"]["teams"]
        while len(teams) < 2:
            stub = f"opposition-{len(teams)}"
            teams.append(stub)
            match["info"]["players"].setdefault(stub, [])

    log.info(
        "loaded %s: %d matches, %d deliveries, %d commentary texts (%s)",
        path, len(matches), stats["rows"], len(reference_texts), dict(stats),
    )
    return matches, reference_texts


def audit_alignment(summary: dict, max_mismatch_rate: float = 0.2) -> None:
    """Fail loudly when the built dataset smells misaligned (§2 rule 7):
    a high entity-mismatch drop rate means commentary text and structured
    names are not from the same deliveries — almost always a column-map or
    over-numbering error, not genuinely dirty data."""
    cleaning = summary.get("cleaning", {})
    total = max(cleaning.get("input_rows", 0), 1)
    mismatches = cleaning.get("dropped_entity_mismatch", 0)
    rate = mismatches / total
    if rate > max_mismatch_rate:
        raise ColumnMapError(
            f"alignment audit FAILED: {mismatches}/{total} rows "
            f"({rate:.0%}) dropped as entity mismatches (threshold "
            f"{max_mismatch_rate:.0%}). Check source.kaggle_commentary.columns "
            "and the over/ball numbering convention before trusting this data."
        )
    log.info("alignment audit passed: %.1f%% entity mismatches", rate * 100)
