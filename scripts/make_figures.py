#!/usr/bin/env python3
"""Generate the four paper figures (§7) from run directories, ACL column width
(3.3 in), PDF + PNG. Figures plot only real runs — systems that have not been
executed simply do not appear (never imputed, §2 rule 7).

1. faithfulness_vs_diversity: tradeoff across decoding settings, per system
2. excitement_calibration:    salience x tension vs excitement, rho annotated
3. hallucination_by_slot:     violations per 100 generations, by slot, per system
4. latency_vs_faithfulness:   RQ3 — bubble area encodes model size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cricket_commentary.eval.base import spearman
from cricket_commentary.eval.excitement import excitement_scores
from cricket_commentary.eval.report import load_all_runs
from cricket_commentary.utils.io import read_jsonl

# fixed categorical slot per system (dataviz palette, light mode) — color
# follows the system identity, never the plot order
SYSTEM_COLORS = {
    "template": "#2a78d6",
    "fewshot": "#1baf7a",
    "finetune_seq2seq": "#eda100",
    "finetune_causal": "#008300",
}
FALLBACK_COLOR = "#4a3aa7"

# param counts for the RQ3 bubble scale
PARAMS = {
    "google/flan-t5-base": 2.5e8,
    "Qwen/Qwen2.5-0.5B-Instruct": 5.0e8,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5e9,
    "Qwen/Qwen2.5-7B-Instruct": 7.0e9,
}

ACL_WIDTH = 3.3

# fixed categorical slot order for per-run coloring (figure 2), dataviz palette
RUN_SLOT_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

_SEED_TAG = __import__("re").compile(r"_seed\d+$")


def dedup_seed_runs(runs: list[dict]) -> list[dict]:
    """Keep one run per seed group (name minus _seed<N>); seeded repeats are
    near-identical points that only overplot — tables carry the mean ± std."""
    seen: set[str] = set()
    out = []
    for run in runs:
        key = _SEED_TAG.sub("", run["run"].get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(run)
    return out


def _style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5,
        "figure.dpi": 120,
    })


def _save(fig, out_dir: Path, name: str):
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{name}.pdf/.png")


def _color(run) -> str:
    return SYSTEM_COLORS.get(run["run"].get("system"), FALLBACK_COLOR)


def _faith(run):
    return run["metrics"].get("faithfulness", {}).get("faithfulness_score")


def _run_config(run) -> dict:
    cfg_path = Path(run["run_dir"]) / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("experiment", {})


def fig_faithfulness_vs_diversity(runs, out_dir):
    fig, ax = plt.subplots(figsize=(ACL_WIDTH, 2.5))
    by_system: dict[str, list] = {}
    for run in dedup_seed_runs(runs):
        faith = _faith(run)
        dist2 = run["metrics"].get("diversity", {}).get("distinct_2")
        if faith is None or dist2 is None:
            continue
        decoding = _run_config(run).get("decoding", {})
        label = ("greedy" if decoding.get("strategy", "greedy") == "greedy"
                 else f"T={decoding.get('temperature', '?')}")
        by_system.setdefault(run["run"]["system"], []).append((dist2, faith, label, _color(run)))
    for system, points in sorted(by_system.items()):
        points.sort()
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        color = points[0][3]
        ax.plot(xs, ys, "-o", color=color, markersize=4, linewidth=1.2, label=system)
        for x, y, label, _ in points:
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(3, 3),
                        fontsize=6, color="#555555")
    ax.set_xlabel("distinct-2 (diversity)")
    ax.set_ylabel("faithfulness (1 − halluc. rate)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, out_dir, "faithfulness_vs_diversity")


def fig_excitement_calibration(runs, out_dir, max_points=300):
    fig, ax = plt.subplots(figsize=(ACL_WIDTH, 2.7))
    rng = np.random.default_rng(13)
    plotted = False
    for run_idx, run in enumerate(dedup_seed_runs(runs)):
        gens_path = Path(run["run_dir"]) / "generations.jsonl"
        data_path = run["run"].get("data")
        split = run["run"].get("split", "test")
        if not gens_path.exists() or not data_path or not Path(data_path).exists():
            print(f"  skip calibration points for {run['run'].get('name')} "
                  f"(missing generations or dataset {data_path})")
            continue
        gens = read_jsonl(gens_path)
        rows = read_jsonl(data_path)
        if split != "all":
            rows = [r for r in rows if r["split"] == split]
        if len(rows) != len(gens) or any(
            g["match_id"] != r["record"]["match_id"] for g, r in zip(gens[:5], rows[:5])
        ):
            print(f"  skip {run['run'].get('name')}: rows/generations misaligned")
            continue
        signal = np.array([r["features"]["event_salience"] * r["features"]["tension"] for r in rows])
        scores, _ = excitement_scores([g["generation"] for g in gens], {"mode": "heuristic"})
        scores = np.array(scores)
        idx = rng.choice(len(rows), size=min(max_points, len(rows)), replace=False)
        # annotate rho recomputed from the SAME heuristic scores plotted here
        # (the stored calibration rho may come from model+heuristic mode)
        rho = spearman(scores.tolist(), signal.tolist())
        label = run["run"]["name"] + (f" (ρ={rho:.2f})" if isinstance(rho, float) else "")
        jitter = rng.normal(0, 0.006, size=len(idx))
        # color per RUN here (fixed slot order): several runs can share one
        # system type and identity must stay distinguishable
        run_color = RUN_SLOT_COLORS[run_idx % len(RUN_SLOT_COLORS)]
        ax.scatter(signal[idx], scores[idx] + jitter, s=6, alpha=0.45,
                   color=run_color, label=label, edgecolors="none")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no runs with recomputable per-ball scores",
                ha="center", va="center", transform=ax.transAxes, color="#777777")
    ax.set_xlabel("event salience × tension")
    ax.set_ylabel("excitement score (heuristic)")
    ax.legend(frameon=False, loc="upper left", markerscale=2)
    _save(fig, out_dir, "excitement_calibration")


def fig_hallucination_by_slot(runs, out_dir):
    fig, ax = plt.subplots(figsize=(ACL_WIDTH, 2.5))
    systems, slot_data = [], {}
    seen_names: dict[str, int] = {}
    for run in dedup_seed_runs(runs):
        faith = run["metrics"].get("faithfulness", {})
        by_type = faith.get("violations_by_type")
        n = faith.get("n")
        if by_type is None or not n:
            continue
        name = run["run"]["name"]
        # runs sharing a name (untagged seed repeats) must not shadow each other
        seen_names[name] = seen_names.get(name, 0) + 1
        if seen_names[name] > 1:
            name = f"{name}#{seen_names[name]}"
        systems.append((name, _color(run)))
        for slot, count in by_type.items():
            slot_data.setdefault(slot, {})[name] = 100.0 * count / n
    slots = sorted(slot_data)
    if not slots:
        slots = ["(no violations in any run)"]
    x = np.arange(len(slots))
    width = 0.8 / max(len(systems), 1)
    for i, (name, color) in enumerate(systems):
        heights = [slot_data.get(slot, {}).get(name, 0.0) for slot in slots]
        ax.bar(x + i * width, heights, width * 0.9, color=color, label=name)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(slots, rotation=30, ha="right")
    ax.set_ylabel("violations / 100 generations")
    if systems:
        ax.legend(frameon=False)
    _save(fig, out_dir, "hallucination_by_slot")


def fig_latency_vs_faithfulness(runs, out_dir):
    fig, ax = plt.subplots(figsize=(ACL_WIDTH, 2.5))
    seen_labels = set()
    for i, run in enumerate(dedup_seed_runs(runs)):
        faith = _faith(run)
        lat = run["metrics"].get("latency", {})
        p50 = lat.get("p50_ms") if isinstance(lat, dict) else None
        if faith is None or p50 is None:
            continue
        cfg = _run_config(run)
        base = cfg.get("model", {}).get("base", "")
        params = PARAMS.get(base)
        size = 30 * np.sqrt(params / 2.5e8) if params else 12
        label = run["run"]["system"]
        ax.scatter([p50], [faith], s=size, color=_color(run), alpha=0.75,
                   label=label if label not in seen_labels else None)
        seen_labels.add(label)
        # stagger labels so co-located points stay readable
        ax.annotate(run["run"]["name"], (p50, faith), textcoords="offset points",
                    xytext=(5, -4 - 7 * i), fontsize=6, color="#555555")
    ax.axvline(1000, color="#e34948", linewidth=0.8, linestyle="--")
    ax.text(870, 0.45, "1 s/ball (RQ3)", fontsize=6, color="#e34948",
            rotation=90, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per ball (ms, log)")
    ax.set_ylabel("faithfulness")
    ax.set_ylim(-0.02, 1.08)
    # framed legend below the data band so its markers never read as data
    ax.legend(loc="center right", framealpha=0.9, edgecolor="#cccccc")
    _save(fig, out_dir, "latency_vs_faithfulness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--include-mini", action="store_true")
    parser.add_argument("--include-smoke", action="store_true",
                        help="include smoke_* machinery-validation runs")
    args = parser.parse_args()

    _style()
    runs = load_all_runs(args.runs)
    if not args.include_mini:
        runs = [r for r in runs if not r["run"].get("name", "").endswith("_mini")]
    if not args.include_smoke:
        runs = [r for r in runs if not r["run"].get("name", "").startswith("smoke_")]
    if not runs:
        raise SystemExit(f"no runs under {args.runs}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_faithfulness_vs_diversity(runs, out_dir)
    fig_excitement_calibration(runs, out_dir)
    fig_hallucination_by_slot(runs, out_dir)
    fig_latency_vs_faithfulness(runs, out_dir)


if __name__ == "__main__":
    main()
