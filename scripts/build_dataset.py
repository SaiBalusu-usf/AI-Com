#!/usr/bin/env python3
"""Build data/processed/dataset.jsonl (and/or the committed fixture) from the
source configured in configs/data.yaml.

Usage:
    python scripts/build_dataset.py --config configs/data.yaml
    python scripts/build_dataset.py --config configs/data.yaml --fixture-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cricket_commentary.data.dataset import build_rows, stratified_fixture
from cricket_commentary.data.synthetic import simulate_matches
from cricket_commentary.utils.config import load_config
from cricket_commentary.utils.io import write_jsonl
from cricket_commentary.utils.logging import get_logger
from cricket_commentary.utils.seeding import set_seed

log = get_logger("build_dataset")


def load_cricsheet_matches(raw_dir: Path, max_matches: int) -> dict[str, dict]:
    paths = sorted(raw_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no Cricsheet match JSON in {raw_dir}. Run `make download` on a "
            "machine with access to cricsheet.org (blocked in the build "
            "environment — see PLAN.md), or set source.kind=synthetic."
        )
    matches: dict[str, dict] = {}
    skipped = 0
    for path in paths:
        if len(matches) >= max_matches:
            break
        with open(path) as f:
            match = json.load(f)
        match_type = (match.get("info") or {}).get("match_type", "")
        if match_type not in ("T20", "IT20"):
            skipped += 1
            continue
        matches[path.stem] = match
    log.info("loaded %d matches from %s (skipped %d non-T20)", len(matches), raw_dir, skipped)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    if not args.fixture_only:
        kind = cfg["source"]["kind"]
        if kind == "cricsheet":
            matches = load_cricsheet_matches(
                Path(cfg["paths"]["raw_dir"]),
                int(cfg["source"]["cricsheet"]["max_matches"]),
            )
        elif kind == "synthetic":
            synth = cfg["source"]["synthetic"]
            matches = simulate_matches(
                int(cfg["seed"]), int(synth["n_matches"]), int(synth["overs_per_innings"])
            )
            log.info("simulated %d synthetic matches (seed=%s)", len(matches), cfg["seed"])
        else:
            raise ValueError(f"unknown source.kind: {kind}")

        rows, summary = build_rows(matches, cfg)
        out = Path(cfg["paths"]["processed"])
        n = write_jsonl(out, rows)
        stats_path = out.with_suffix(".stats.json")
        stats_path.write_text(json.dumps(summary, indent=2))
        log.info("wrote %d rows -> %s (stats: %s)", n, out, stats_path)
        log.info("summary: %s", json.dumps(summary))

    # fixture: independent seed + small match count, stratified to n_rows
    fx = cfg["fixture"]
    fixture_matches = simulate_matches(int(fx["seed"]), n_matches=6, overs=20)
    fx_cfg = dict(cfg, seed=int(fx["seed"]))
    fx_cfg["source"] = dict(cfg["source"], kind="synthetic")
    rows, summary = build_rows(fixture_matches, fx_cfg)
    fixture_rows = stratified_fixture(rows, int(fx["n_rows"]), seed=int(fx["seed"]))
    n = write_jsonl(cfg["paths"]["fixture"], fixture_rows)
    log.info(
        "wrote %d fixture rows -> %s (from %d candidate rows; splits present: %s)",
        n,
        cfg["paths"]["fixture"],
        summary["rows"],
        sorted({r["split"] for r in fixture_rows}),
    )


if __name__ == "__main__":
    main()
