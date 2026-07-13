"""Tiny JSONL helpers used across the pipeline and the harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({err})") from err
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # newline="\n" keeps output byte-identical across platforms — Windows'
    # default CRLF translation would break the committed fixture's
    # regeneration-determinism guarantee
    # explicit utf-8: Windows' cp1252 default cannot encode real commentary
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
