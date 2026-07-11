#!/usr/bin/env python3
"""Download raw Cricsheet ball-by-ball JSON (network required).

The build environment's egress policy blocks cricsheet.org (verified
2026-07-11; see PLAN.md). Run this on a machine with normal network access:

    python scripts/download_data.py --config configs/data.yaml

Cricsheet data is released under the Open Data Commons Attribution licence
(ODC-BY) — the licence text is linked from every download page; this script
records the URL + retrieval date into data/raw/PROVENANCE.json and the human
confirms the licence in data/DATASET_CARD.md (decision D1 covers the separate
parallel-commentary corpus, which this script does NOT fetch).
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

from cricket_commentary.utils.config import load_config
from cricket_commentary.utils.logging import get_logger

log = get_logger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="only test reachability")
    args = parser.parse_args()

    cfg = load_config(args.config)
    url = cfg["source"]["cricsheet"]["download_url"]
    raw_dir = Path(cfg["paths"]["raw_dir"])

    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        head.raise_for_status()
    except requests.RequestException as err:
        log.error(
            "cricsheet.org unreachable (%s). If you are inside the restricted "
            "build environment this is the documented egress block — run this "
            "script on your own machine instead.", err,
        )
        sys.exit(2)

    if args.check:
        log.info("reachable: %s", url)
        return

    log.info("downloading %s ...", url)
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        zf.extractall(raw_dir, members=names)
    provenance = {
        "url": url,
        "retrieved": datetime.date.today().isoformat(),
        "files": len(names),
        "license": "ODC-BY (confirm on cricsheet.org/register/ and record in DATASET_CARD.md)",
    }
    (raw_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    log.info("extracted %d match files -> %s; provenance recorded", len(names), raw_dir)


if __name__ == "__main__":
    main()
