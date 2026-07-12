"""Shared generation loop (§3): every system produces its outputs through the
same code path, and every run directory stores the same generations.jsonl
shape — one line per ball with input, reference, and generation."""

from __future__ import annotations

from typing import Iterable, Protocol

from tqdm import tqdm


class Generator(Protocol):
    def generate(self, row: dict) -> str: ...


def generate_all(generator: Generator, rows: list[dict], show_progress: bool = True) -> list[str]:
    iterator = tqdm(rows, desc="generate", disable=not show_progress)
    return [generator.generate(row) for row in iterator]


def generations_payload(rows: list[dict], texts: list[str]) -> Iterable[dict]:
    if len(rows) != len(texts):
        raise ValueError("rows and generations must be aligned")
    for row, text in zip(rows, texts):
        record = row["record"]
        yield {
            "match_id": record["match_id"],
            "innings": record["innings"],
            "over": record["over"],
            "ball": record["ball"],
            "linearized_input": row["linearized_input"],
            "reference": row["target_commentary"],
            "generation": text,
        }
