"""Commentary-target cleaning (§4): HTML stripping, dedup, length filter, and
regex entity-validation against the structured record.

These functions are source-agnostic: they run over synthetic references today
and over the real parallel corpus once D1 (dataset choice) is resolved. Every
drop is counted so the dataset card can report cleaning stats honestly.
"""

from __future__ import annotations

import html
import re
from collections import Counter

from .ingest import BallRecord

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html.unescape(text))).strip()


def token_count(text: str) -> int:
    return len(text.split())


def _surname(name: str) -> str:
    return name.split()[-1] if name.strip() else ""


def entity_mismatch(text: str, record: BallRecord) -> bool:
    """True when the commentary names a squad member who is NOT involved in
    this ball while never naming the actual batter or bowler — the §4
    misalignment signal (commentary row joined to the wrong ball)."""
    involved = {
        _surname(n)
        for n in (
            [record.batter, record.bowler, record.non_striker, record.player_out]
            + record.fielders
        )
        if n
    }
    squad_surnames = {_surname(n) for n in record.squad} - involved
    words = set(re.findall(r"[A-Z][a-z]+", text))
    mentions_involved = bool(words & involved)
    mentions_others = bool(words & squad_surnames)
    return mentions_others and not mentions_involved


def clean_targets(
    rows: list[dict],
    records: list[BallRecord],
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[dict], Counter]:
    """Filter (row, record) pairs; returns surviving rows + drop-reason counts.

    ``rows`` are pipeline rows with a ``target_commentary`` field, aligned
    index-for-index with ``records``.
    """
    if len(rows) != len(records):
        raise ValueError("rows and records must be aligned")
    stats: Counter = Counter(input_rows=len(rows))
    seen: set[str] = set()
    kept: list[dict] = []
    for row, record in zip(rows, records):
        text = strip_html(row["target_commentary"])
        if not text:
            stats["dropped_empty"] += 1
            continue
        n_tokens = token_count(text)
        if n_tokens < min_tokens:
            stats["dropped_too_short"] += 1
            continue
        if n_tokens > max_tokens:
            stats["dropped_too_long"] += 1
            continue
        if entity_mismatch(text, record):
            stats["dropped_entity_mismatch"] += 1
            continue
        key = text.lower()
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        row = dict(row, target_commentary=text)
        kept.append(row)
    stats["kept"] = len(kept)
    return kept, stats
