#!/usr/bin/env python3
"""Run one config-driven experiment end-to-end: load data, build the system,
generate, evaluate through the shared harness, and write a run directory
(results/runs/<UTC timestamp>_<name>[_<tag>]/ with resolved config copy,
generations.jsonl, metrics.json, report.md, run.log).

Usage:
    python scripts/run_experiment.py --config configs/baseline_template.yaml
    python scripts/run_experiment.py --config ... --data data/fixtures/mini.jsonl --tag mini
    python scripts/run_experiment.py --config ... --set decoding.temperature=1.0 --tag t10
    python scripts/run_experiment.py --config ... --plan-only
    python scripts/run_experiment.py --eval-only --run results/runs/<dir>

Long-run policy (§9 rule 5): if the estimated generation time exceeds 30
minutes, the run aborts with the command + estimate unless --yes is passed.
Paid API backends additionally require --confirm-spend.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

from cricket_commentary.eval.harness import evaluate
from cricket_commentary.eval.report import git_commit, write_run_report
from cricket_commentary.models.fewshot import GeneratorUnavailable
from cricket_commentary.models.generate import generate_all, generations_payload
from cricket_commentary.utils.config import apply_overrides, load_config, require
from cricket_commentary.utils.io import read_jsonl, write_jsonl
from cricket_commentary.utils.logging import add_run_file_handler, get_logger
from cricket_commentary.utils.seeding import set_seed

log = get_logger("run_experiment")

LONG_RUN_SECONDS = 30 * 60


def build_generator(cfg: dict, rows_train: list[dict], allow_paid_api: bool):
    system = require(cfg, "system")
    seed = int(cfg["seed"])
    gen_cfg = cfg.get("generator", {})
    decoding = cfg.get("decoding", {})
    if system == "template":
        from cricket_commentary.models.template import build

        return build(gen_cfg, seed)
    if system == "fewshot":
        from cricket_commentary.models.fewshot import build

        return build(gen_cfg, seed, rows_train, decoding, allow_paid_api)
    if system in ("finetune_seq2seq", "finetune_causal"):
        from cricket_commentary.models.finetune import build_inference

        return build_inference(cfg)
    raise ValueError(f"unknown system: {system}")


def select_rows(dataset_path: str, split: str) -> list[dict]:
    rows = read_jsonl(dataset_path)
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {dataset_path}")
    return rows


def estimate_seconds(generator, rows: list[dict], probe: int = 3) -> float:
    start = time.perf_counter()
    for row in rows[:probe]:
        generator.generate(row)
    per_call = (time.perf_counter() - start) / max(min(probe, len(rows)), 1)
    return per_call * len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="experiment YAML")
    parser.add_argument("--data", help="override dataset path (e.g. the fixture)")
    parser.add_argument("--split", help="train|val|test|all (default: config value; 'all' when --data is given)")
    parser.add_argument("--tag", default="", help="suffix for the run dir name")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="dotted config override key.sub=value (repeatable)")
    parser.add_argument("--plan-only", action="store_true",
                        help="print what would run + time estimate, then exit")
    parser.add_argument("--yes", action="store_true",
                        help="approve a run estimated to exceed 30 minutes (§9.5)")
    parser.add_argument("--confirm-spend", action="store_true",
                        help="approve paid API backends (§9.5)")
    parser.add_argument("--eval-only", action="store_true",
                        help="re-evaluate an existing run directory")
    parser.add_argument("--run", help="run dir for --eval-only")
    args = parser.parse_args()

    if args.eval_only:
        return eval_only(args)
    if not args.config:
        parser.error("--config is required unless --eval-only")

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed = int(require(cfg, "seed"))
    set_seed(seed)
    eval_cfg = load_config(require(cfg, "eval_config"))

    dataset_path = args.data or require(cfg, "data.dataset")
    split = args.split or ("all" if args.data else cfg["data"].get("split", "test"))
    rows = select_rows(dataset_path, split)
    train_rows = (
        select_rows(dataset_path, "train")
        if require(cfg, "system") != "template"
        else []
    )
    if "linearization" in cfg:  # ablations A1/A2: re-linearize at runtime
        from cricket_commentary.data.dataset import relinearize_rows

        rows = relinearize_rows(rows, cfg["linearization"])
        train_rows = relinearize_rows(train_rows, cfg["linearization"]) if train_rows else []
        log.info("re-linearized inputs with %s", cfg["linearization"])

    try:
        generator = build_generator(cfg, train_rows, args.confirm_spend)
    except GeneratorUnavailable as err:
        # environment blockage, not a code bug (those propagate with traceback)
        log.error("generator for %s could not be built: %s", cfg["name"], err)
        log.error("this blocked attempt should be logged in EXPERIMENTS.md (§9.3)")
        if args.plan_only:
            print(f"plan: run={cfg['name']} system={cfg['system']} rows={len(rows)} "
                  f"split={split} — NOT RUNNABLE HERE: {err}")
            return
        sys.exit(4)

    # the harness generates everything twice (once for outputs, once timed
    # inside the latency pass) — the estimate and any spend approval must
    # reflect the true call count
    warmup = int(eval_cfg.get("latency", {}).get("warmup_calls", 5))
    est = estimate_seconds(generator, rows) * 2 + estimate_seconds(generator, rows[:1]) * warmup
    plan = (
        f"run={cfg['name']} system={cfg['system']} rows={len(rows)} "
        f"split={split} data={dataset_path} seed={seed} "
        f"estimated_total_generation={est/60:.1f} min "
        f"(~{2 * len(rows) + warmup} generate calls incl. latency pass)"
    )
    log.info(plan)
    if args.plan_only:
        print(plan)
        return
    if est > LONG_RUN_SECONDS and not args.yes:
        print(f"REFUSING long run without approval (§9.5): {plan}\n"
              f"re-run with --yes once approved")
        sys.exit(3)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = Path("results/runs") / f"{stamp}_{cfg['name']}{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    add_run_file_handler(run_dir)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({"experiment": cfg, "eval": eval_cfg}, f, sort_keys=False)

    log.info("generating %d commentaries ...", len(rows))
    generations = generate_all(generator, rows)
    write_jsonl(run_dir / "generations.jsonl", generations_payload(rows, generations))

    log.info("evaluating through the shared harness ...")
    metrics = evaluate(rows, generations, eval_cfg, generate_fn=generator.generate)
    run_meta = {
        "name": cfg["name"] + tag,
        "system": cfg["system"],
        "config_path": args.config,
        "overrides": args.overrides,
        "data": dataset_path,
        "split": split,
        "n_rows": len(rows),
        "seed": seed,
        "timestamp": stamp,
        "git_commit": git_commit("."),
    }
    report_path = write_run_report(run_dir, run_meta, metrics)
    log.info("run complete: %s", run_dir)
    faith = metrics.get("faithfulness", {})
    print(
        f"DONE {run_dir}\n"
        f"  faithfulness={faith.get('faithfulness_score')} "
        f"hallucination={faith.get('hallucination_rate')} "
        f"omission={faith.get('omission_rate')}\n"
        f"  report: {report_path}"
    )


def eval_only(args) -> None:
    if not args.run:
        raise SystemExit("--eval-only requires --run <run_dir>")
    run_dir = Path(args.run)
    stored = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg, eval_cfg = stored["experiment"], stored["eval"]
    set_seed(int(cfg["seed"]))

    old = json.loads((run_dir / "metrics.json").read_text())
    stored_meta = old["run"]
    # the run's recorded provenance, not the config defaults — the run may
    # have been launched with --data/--split overrides (e.g. the fixture)
    dataset_path = args.data or stored_meta.get("data") or cfg["data"]["dataset"]
    split = args.split or stored_meta.get("split") or cfg["data"].get("split", "test")
    rows = select_rows(dataset_path, split)
    if "linearization" in cfg:
        from cricket_commentary.data.dataset import relinearize_rows

        rows = relinearize_rows(rows, cfg["linearization"])
    gens = read_jsonl(run_dir / "generations.jsonl")
    if len(gens) != len(rows) or any(
        g["match_id"] != r["record"]["match_id"]
        or g.get("delivery_seq", r["record"].get("delivery_seq"))
        != r["record"].get("delivery_seq")
        for g, r in zip(gens, rows)
    ):
        raise SystemExit(
            f"stored generations ({len(gens)}) do not align with rows "
            f"({len(rows)}) from {dataset_path}[{split}]"
        )
    metrics = evaluate(rows, [g["generation"] for g in gens], eval_cfg)
    old_latency = old.get("metrics", {}).get("latency", {})
    if isinstance(old_latency, dict) and "p50_ms" in old_latency:
        # re-evaluation cannot re-time generation; keep the measured numbers
        metrics["latency"] = dict(
            old_latency, note="carried from the original run (eval-only rerun)"
        )
    stored_meta["reevaluated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report_path = write_run_report(run_dir, stored_meta, metrics)
    print(f"re-evaluated {run_dir}\n  report: {report_path}")


if __name__ == "__main__":
    main()
