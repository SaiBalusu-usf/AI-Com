#!/usr/bin/env python3
"""Train M1 (flan-t5-base LoRA) or M2 (Qwen QLoRA) per config.

Usage:
    python scripts/train_model.py --config configs/finetune_t5.yaml --smoke
    python scripts/train_model.py --config configs/finetune_t5.yaml --plan-only
    python scripts/train_model.py --config configs/finetune_t5.yaml --yes

--smoke trains 50 steps on the committed fixture (train split) — the Phase 4
gate. Full runs print their plan and REQUIRE --yes (§9 rule 5: ask before any
run that may exceed 30 minutes). After training, evaluate the adapter through
the shared harness:
    python scripts/run_experiment.py --config <same config> \
        --set model.adapter_dir=<run_dir>/adapter
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import yaml

from cricket_commentary.models.fewshot import GeneratorUnavailable
from cricket_commentary.models.finetune import resolve_device_plan, train
from cricket_commentary.utils.config import apply_overrides, load_config
from cricket_commentary.utils.io import read_jsonl
from cricket_commentary.utils.logging import add_run_file_handler, get_logger
from cricket_commentary.utils.seeding import set_seed

log = get_logger("train_model")

SMOKE_STEPS = 50
FIXTURE = "data/fixtures/mini.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_STEPS} steps on the fixture (phase-4 gate)")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--yes", action="store_true", help="approve a full run (§9.5)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(int(cfg["seed"]))

    plan = resolve_device_plan(cfg)
    dataset = FIXTURE if args.smoke else cfg["data"]["dataset"]
    plan_line = (
        f"train {cfg['name']} ({cfg['system']}) base={cfg['model']['base']} "
        f"data={dataset} smoke={args.smoke} device_plan={plan}"
    )
    print(plan_line)
    if args.plan_only:
        if not plan.get("available"):
            print(f"NOT RUNNABLE HERE: {plan['reason']}")
        else:
            print("full training on CPU will exceed 30 min — launch with --yes once approved (§9.5)"
                  if not plan.get("cuda") else "CUDA available — expect minutes-to-hours depending on data size")
        return
    if not plan.get("available"):
        log.error("cannot train: %s", plan["reason"])
        sys.exit(4)
    if not args.smoke and not args.yes:
        print("REFUSING full training run without --yes (§9.5). Plan:\n  " + plan_line)
        sys.exit(3)

    rows = read_jsonl(dataset)
    rows_train = [r for r in rows if r["split"] == "train"]
    rows_val = [r for r in rows if r["split"] == "val"] or rows_train[:32]

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("results/runs") / f"{stamp}_{cfg['name']}_{'smoke' if args.smoke else 'train'}"
    run_dir.mkdir(parents=True, exist_ok=False)
    add_run_file_handler(run_dir)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({"experiment": cfg}, f, sort_keys=False)

    try:
        summary = train(
            cfg, rows_train, rows_val, run_dir,
            max_steps=SMOKE_STEPS if args.smoke else None,
        )
    except GeneratorUnavailable as err:
        log.error("training unavailable: %s", err)
        sys.exit(4)

    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE {run_dir}\n  adapter: {summary['adapter_dir']}\n"
          f"  train_loss={summary['train_loss']:.4f} steps={summary['steps']}\n"
          f"  evaluate with: python scripts/run_experiment.py --config {args.config} "
          f"--set model.adapter_dir={summary['adapter_dir']}")


if __name__ == "__main__":
    main()
