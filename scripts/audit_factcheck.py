#!/usr/bin/env python3
"""Measure the fact checker's OWN accuracy against human judgment (§5.1).

The checker's adversarial unit tests prove behaviour on curated sentences;
the paper additionally needs its precision/recall on the systems' real
outputs. Two-step protocol:

1. Sample a stratified annotation sheet from a run directory:
       python scripts/audit_factcheck.py --sample 100 \\
           --run results/runs/<dir> --out results/annotation/<name>.csv
   Fill the two human columns per row:
       human_hallucination: y if the text asserts something false about the
           ball (against linearized_input), else n
       human_omission: y if a wicket/four/six on the ball goes unmentioned,
           else n (leave n for non-salient balls)

2. Score checker-vs-human agreement:
       python scripts/audit_factcheck.py --score results/annotation/<name>.csv
   Reports per-task precision/recall/F1, accuracy, and Cohen's kappa, and
   writes <name>.scores.json next to the sheet.

Stratification oversamples rows the checker flagged (violations/omissions are
rare but are exactly where checker errors matter); the sheet records the
stratum so scoring can also report per-stratum agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cricket_commentary.eval.factcheck import check_generation
from cricket_commentary.utils.io import read_jsonl


def _checker_verdicts(rows: list[dict], gens: list[dict]) -> list[dict]:
    out = []
    for row, gen in zip(rows, gens):
        res = check_generation(gen["generation"], row["record"])
        out.append(
            {
                "match_id": row["record"]["match_id"],
                "innings": row["record"]["innings"],
                "over": row["record"]["over"],
                "delivery_seq": row["record"].get("delivery_seq", 1),
                "linearized_input": row["linearized_input"],
                "generation": gen["generation"],
                "checker_hallucination": "y" if res.violations else "n",
                "checker_omission": "y" if res.omissions else "n",
                "checker_detail": "; ".join(
                    f"{v['slot']}:{v['claimed']}vs{v['actual']}" for v in res.violations
                ) + ("; " if res.violations and res.omissions else "") + "; ".join(res.omissions),
            }
        )
    return out


def sample(args) -> None:
    run_dir = Path(args.run)
    meta = json.loads((run_dir / "metrics.json").read_text())["run"]
    rows = read_jsonl(meta["data"])
    if meta.get("split", "all") != "all":
        rows = [r for r in rows if r["split"] == meta["split"]]
    gens = read_jsonl(run_dir / "generations.jsonl")
    if len(rows) != len(gens):
        raise SystemExit("run generations do not align with its recorded dataset/split")

    verdicts = _checker_verdicts(rows, gens)
    rng = np.random.default_rng(int(args.seed))
    strata = {
        "flagged_hallucination": [i for i, v in enumerate(verdicts) if v["checker_hallucination"] == "y"],
        "flagged_omission": [i for i, v in enumerate(verdicts)
                             if v["checker_omission"] == "y" and v["checker_hallucination"] == "n"],
        "clean": [i for i, v in enumerate(verdicts)
                  if v["checker_hallucination"] == "n" and v["checker_omission"] == "n"],
    }
    # up to a third from each flagged stratum, remainder from clean
    per_flagged = max(args.sample // 3, 1)
    chosen: list[int] = []
    for name in ("flagged_hallucination", "flagged_omission"):
        pool = strata[name]
        take = min(per_flagged, len(pool))
        chosen += rng.choice(pool, size=take, replace=False).tolist() if take else []
    remaining = [i for i in strata["clean"] if i not in set(chosen)]
    take = min(args.sample - len(chosen), len(remaining))
    chosen += rng.choice(remaining, size=take, replace=False).tolist()
    chosen = sorted(set(chosen))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(verdicts[0].keys()) + ["stratum", "human_hallucination", "human_omission", "human_notes"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in chosen:
            stratum = next(name for name, pool in strata.items() if i in set(pool))
            writer.writerow({**verdicts[i], "stratum": stratum,
                             "human_hallucination": "", "human_omission": "", "human_notes": ""})
    print(f"wrote {len(chosen)} rows -> {out}\n"
          f"strata: " + ", ".join(f"{k}={len(v)} available" for k, v in strata.items()) +
          "\nfill human_hallucination / human_omission (y|n), then --score")


def _binary_scores(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(not p and g for p, g in zip(pred, gold))
    tn = sum(not p and not g for p, g in zip(pred, gold))
    n = len(pred)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and precision + recall else None)
    accuracy = (tp + tn) / n
    # Cohen's kappa
    p_yes = (tp + fp) / n * (tp + fn) / n
    p_no = (tn + fn) / n * (tn + fp) / n
    p_e = p_yes + p_no
    kappa = (accuracy - p_e) / (1 - p_e) if p_e < 1 else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "kappa": kappa}


def score(args) -> None:
    path = Path(args.score)
    with open(path, newline="") as f:
        sheet = list(csv.DictReader(f))
    unlabelled = [r for r in sheet if r["human_hallucination"].strip().lower() not in ("y", "n")
                  or r["human_omission"].strip().lower() not in ("y", "n")]
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled)} rows lack y/n labels in human_hallucination/"
            "human_omission — finish annotating before scoring"
        )

    results = {}
    for task in ("hallucination", "omission"):
        pred = [r[f"checker_{task}"].strip().lower() == "y" for r in sheet]
        gold = [r[f"human_{task}"].strip().lower() == "y" for r in sheet]
        results[task] = _binary_scores(pred, gold)
        by_stratum = {}
        for stratum in sorted({r["stratum"] for r in sheet}):
            idx = [i for i, r in enumerate(sheet) if r["stratum"] == stratum]
            by_stratum[stratum] = _binary_scores([pred[i] for i in idx], [gold[i] for i in idx])
        results[task]["by_stratum"] = by_stratum

    out = path.with_suffix(".scores.json")
    out.write_text(json.dumps(results, indent=2))
    for task, r in results.items():
        print(f"{task}: precision={r['precision']} recall={r['recall']} "
              f"f1={r['f1']} kappa={r['kappa']} (n={r['n']})")
    print(f"written -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, help="sheet size to draw")
    parser.add_argument("--run", help="run directory to sample from")
    parser.add_argument("--out", help="annotation CSV path")
    parser.add_argument("--seed", default=13)
    parser.add_argument("--score", help="filled annotation CSV to score")
    args = parser.parse_args()
    if args.score:
        score(args)
    elif args.sample:
        if not (args.run and args.out):
            parser.error("--sample needs --run and --out")
        sample(args)
    else:
        parser.error("use --sample ... or --score ...")


if __name__ == "__main__":
    main()
