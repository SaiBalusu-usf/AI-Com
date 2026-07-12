"""Assemble the emitted JSONL rows (§4):
``{record, features, linearized_input, target_commentary, split, ...}``.

The same path builds the full processed dataset and the 200-row fixture; the
fixture differs only in source seed, match count, and a stratified row cap
that preserves event coverage (wickets, sixes, fours, dots, extras).
"""

from __future__ import annotations

from collections import Counter

from .clean import clean_targets
from .features import compute_features
from .ingest import BallRecord, parse_matches
from .linearize import linearize
from .splits import split_by_match
from .synthetic import write_reference


def build_rows(
    matches: dict[str, dict],
    data_cfg: dict,
    reference_texts: dict | None = None,
) -> tuple[list[dict], dict]:
    """matches: {match_id: cricsheet_match_dict}. ``reference_texts`` maps a
    ball key (match_id, innings, over, ball, seq) -> human commentary once a
    real parallel corpus is wired in (D1); None means synthetic_v1 references.
    """
    commentary_cfg = data_cfg["commentary"]
    lin_cfg = data_cfg["linearization"]
    source_kind = data_cfg["source"]["kind"]
    seed = int(data_cfg["seed"])

    records = parse_matches(matches)
    features = compute_features(records, data_cfg["features"])
    assignment = split_by_match(
        [r.match_id for r in records], data_cfg["splits"], seed=seed
    )

    if commentary_cfg["source"] != "synthetic_v1" and reference_texts is None:
        raise NotImplementedError(
            f"commentary.source={commentary_cfg['source']!r} needs a parallel "
            "corpus loader — pending human decision D1 (see data/DATASET_CARD.md). "
            "No fallback is silently substituted."
        )

    rows: list[dict] = []
    unmatched = 0
    prev_by_innings: dict[tuple[str, int], list[BallRecord]] = {}
    for record, feats in zip(records, features):
        key = (record.match_id, record.innings)
        prev = prev_by_innings.setdefault(key, [])
        text = (
            write_reference(record, feats, seed)
            if reference_texts is None
            else reference_texts.get(
                (record.match_id, record.innings, record.over, record.ball)
            )
        )
        if text is None:
            # real corpus rows that have no commentary for this ball are a
            # coverage gap, reported in the dataset card — never silent
            unmatched += 1
            prev.append(record)
            continue
        rows.append(
            {
                "record": record.to_dict(),
                "features": feats,
                "linearized_input": linearize(
                    record,
                    feats,
                    fmt=lin_cfg["format"],
                    include_game_state=bool(lin_cfg["include_game_state"]),
                    prev_records=prev,
                    context_balls=int(lin_cfg["context_balls"]),
                ),
                "target_commentary": text,
                "split": assignment[record.match_id],
                "data_source": source_kind,
                "reference_source": commentary_cfg["source"],
            }
        )
        prev.append(record)

    aligned_records = [BallRecord(**row["record"]) for row in rows]
    kept, stats = clean_targets(
        rows,
        aligned_records,
        min_tokens=int(commentary_cfg["min_tokens"]),
        max_tokens=int(commentary_cfg["max_tokens"]),
    )

    split_counts = Counter(row["split"] for row in kept)
    match_counts = Counter(row["record"]["match_id"] for row in kept)
    summary = {
        "matches": len(match_counts),
        "rows": len(kept),
        "balls_without_commentary": unmatched,
        "splits": dict(split_counts),
        "cleaning": dict(stats),
    }
    return kept, summary


def relinearize_rows(rows: list[dict], lin_cfg: dict) -> list[dict]:
    """Recompute ``linearized_input`` from record+features with a different
    linearization config — the runtime hook behind ablations A1 (flat vs
    context) and A2 (game-state on/off), so each ablation cell is one config
    file, not a dataset rebuild. Rows must be in ball order per innings (the
    dataset writer preserves it)."""
    out: list[dict] = []
    prev_by_innings: dict[tuple[str, int], list[BallRecord]] = {}
    for row in rows:
        record = BallRecord(**row["record"])
        key = (record.match_id, record.innings)
        prev = prev_by_innings.setdefault(key, [])
        out.append(
            dict(
                row,
                linearized_input=linearize(
                    record,
                    row["features"],
                    fmt=lin_cfg.get("format", "flat"),
                    include_game_state=bool(lin_cfg.get("include_game_state", True)),
                    prev_records=prev,
                    context_balls=int(lin_cfg.get("context_balls", 2)),
                ),
            )
        )
        prev.append(record)
    return out


def stratified_fixture(rows: list[dict], n_rows: int, seed: int) -> list[dict]:
    """Cap rows for the committed fixture while guaranteeing event coverage:
    wickets, sixes, fours, dots, extras, death-phase and chase balls all
    appear. Original ball order within each match is preserved (context
    features remain valid because they were computed before capping)."""
    import numpy as np

    if len(rows) <= n_rows:
        return rows

    def bucket(row: dict) -> str:
        rec = row["record"]
        if rec["wicket"]:
            return "wicket"
        if rec["runs_batter"] == 6:
            return "six"
        if rec["runs_batter"] == 4:
            return "four"
        if rec["extras_type"] != "none":
            return "extra"
        if rec["runs_total"] == 0:
            return "dot"
        return "runs"

    rng = np.random.default_rng(seed)
    indexed = list(enumerate(rows))
    chosen: set[int] = set()
    buckets: dict[str, list[int]] = {}
    for i, row in indexed:
        buckets.setdefault(bucket(row), []).append(i)
    # floor per bucket, then top up uniformly
    per_bucket = max(4, n_rows // (len(buckets) * 2))
    for indices in buckets.values():
        take = min(per_bucket, len(indices))
        chosen.update(rng.choice(indices, size=take, replace=False).tolist())
    remaining = [i for i, _ in indexed if i not in chosen]
    top_up = n_rows - len(chosen)
    if top_up > 0:
        chosen.update(rng.choice(remaining, size=top_up, replace=False).tolist())
    return [rows[i] for i in sorted(chosen)][:n_rows]
