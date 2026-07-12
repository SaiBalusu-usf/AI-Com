"""Per-run reporting (§5.4): metrics.json plus a human-readable report.md.

Every run directory gets a machine-readable ``metrics.json``
(``{"run": ..., "metrics": ...}``) and a ``report.md`` for humans. Metric
availability gaps (§2 rule 7) — subtrees of the form
``{"status": "unavailable", "reason": ...}`` or ``{"status": "disabled"}`` —
are surfaced twice: inline in their metric section and again under a final
"Metrics not computed" heading, so they are impossible to miss.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# "blocked" is written by harness.py when the excitement scorer fails its
# reference validation — it must surface as a gap too, not vanish
_GAP_STATUSES = ("unavailable", "disabled", "blocked")


def _is_gap(value: object) -> bool:
    """True for the not-computed sentinels written by the harness (§2 rule 7)."""
    return isinstance(value, dict) and value.get("status") in _GAP_STATUSES


def _gap_reason(gap: dict) -> str:
    if gap.get("status") == "disabled":
        return "disabled"
    return str(gap.get("reason", "unavailable"))


def _fmt_leaf(value: object) -> str:
    if isinstance(value, dict) and _is_gap(value):
        if value.get("status") == "disabled":
            return "disabled"
        return f"unavailable ({_gap_reason(value)})"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _flatten(value: object, prefix: str = "") -> list[tuple[str, object]]:
    """(dotted key path, leaf) pairs in insertion order; gap dicts stay whole."""
    if isinstance(value, dict) and not _is_gap(value):
        rows: list[tuple[str, object]] = []
        for key, sub in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(sub, path))
        return rows
    return [(prefix, value)]


def _collect_gaps(value: object, prefix: str = "") -> list[tuple[str, str]]:
    """(dotted path, reason) for every unavailable/disabled subtree, in order."""
    if _is_gap(value):
        return [(prefix, _gap_reason(value))]  # type: ignore[arg-type]
    if isinstance(value, dict):
        gaps: list[tuple[str, str]] = []
        for key, sub in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            gaps.extend(_collect_gaps(sub, path))
        return gaps
    return []


def _table(rows: list[tuple[str, str]], header: tuple[str, str]) -> list[str]:
    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines.extend(f"| {key} | {val} |" for key, val in rows)
    return lines


def write_run_report(run_dir: str | Path, run_meta: dict, metrics: dict) -> Path:
    """Write ``<run_dir>/metrics.json`` and ``<run_dir>/report.md``.

    ``run_meta`` carries name, config_path, seed, timestamp and optionally
    git_commit. Returns the report.md path.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"run": run_meta, "metrics": metrics}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    lines: list[str] = [f"# Run report: {run_meta.get('name', run_dir.name)}", ""]
    lines += ["## Run info", ""]
    lines += _table([(str(k), _fmt_leaf(v)) for k, v in run_meta.items()], ("field", "value"))
    lines.append("")

    for key, value in metrics.items():
        lines += [f"## {key}", ""]
        rows = [(path or str(key), _fmt_leaf(leaf)) for path, leaf in _flatten(value)]
        lines += _table(rows, ("metric", "value"))
        lines.append("")

    lines += ["## Metrics not computed", ""]
    gaps = _collect_gaps(metrics)
    if gaps:
        lines += [f"- `{path}`: {reason}" for path, reason in gaps]
    else:
        lines.append("None - all configured metrics were computed.")
    lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def git_commit(repo_dir: str | Path = ".") -> str | None:
    """Short commit hash of ``repo_dir``'s HEAD, or None when unavailable
    (no git binary, not a repo, no commits). Never raises for those cases."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):  # FileNotFoundError is an OSError
        return None
    return proc.stdout.strip() or None


def load_all_runs(runs_root: str | Path) -> list[dict]:
    """Every ``<runs_root>/<dir>/metrics.json`` as
    ``{"run": ..., "metrics": ..., "run_dir": str}``, sorted by run dir name.

    Dirs without a metrics.json are silently skipped; a missing root yields [].
    """
    root = Path(runs_root)
    if not root.exists():
        return []
    runs: list[dict] = []
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            payload = json.load(f)
        runs.append(
            {
                "run": payload.get("run", {}),
                "metrics": payload.get("metrics", {}),
                "run_dir": str(run_dir),
            }
        )
    return runs
