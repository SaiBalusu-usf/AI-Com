"""The single evaluation pipeline every system goes through (§5):
``evaluate(rows, generations, eval_cfg, generate_fn) -> metrics dict``.

Ordering constraint from §5.2: the excitement scorer must be validated on the
HUMAN REFERENCES of the same rows before it is ever applied to generations.
If validation fails (non-positive Spearman), calibration is reported as
blocked — never computed with an invalidated scorer, and never silently
dropped.
"""

from __future__ import annotations

from typing import Callable

from ..utils.logging import get_logger
from . import diversity, excitement, factcheck, latency, surface
from .base import unavailable

log = get_logger("harness")


def evaluate(
    rows: list[dict],
    generations: list[str],
    eval_cfg: dict,
    generate_fn: Callable[[dict], str] | None = None,
) -> dict:
    if len(rows) != len(generations):
        raise ValueError("rows and generations must be aligned")
    if not rows:
        raise ValueError("nothing to evaluate")
    references = [row["target_commentary"] for row in rows]
    metrics: dict = {}

    if eval_cfg["factcheck"].get("enabled", True):
        metrics["faithfulness"] = factcheck.factcheck_metrics(
            rows, generations, eval_cfg["factcheck"]
        )
        # human references themselves omit explicit event words ("into the
        # crowd!" for a six); their rates are the natural baseline systems
        # should be read against, so publish them alongside
        ref_check = factcheck.factcheck_metrics(rows, references, eval_cfg["factcheck"])
        metrics["faithfulness"]["reference_baseline"] = {
            "hallucination_rate": ref_check["hallucination_rate"],
            "omission_rate": ref_check["omission_rate"],
        }

    excite_cfg = eval_cfg["excitement"]
    validation = excitement.validate_on_references(rows, excite_cfg)
    if validation["passed"]:
        calibration = excitement.calibration(rows, generations, excite_cfg)
    else:
        log.warning(
            "excitement scorer FAILED reference validation (rho=%s) — "
            "calibration blocked; iterate on the scorer before trusting it (§5.2)",
            validation["spearman_rho"],
        )
        calibration = {
            "status": "blocked",
            "reason": f"scorer failed reference validation (rho={validation['spearman_rho']})",
        }
    metrics["excitement"] = {
        "reference_validation": validation,
        "calibration": calibration,
    }

    metrics["surface"] = surface.surface_metrics(
        references, generations, eval_cfg["surface"]
    )
    metrics["diversity"] = diversity.diversity_metrics(
        rows, generations, eval_cfg["diversity"], seed=int(eval_cfg["seed"])
    )

    if generate_fn is not None:
        metrics["latency"] = latency.measure_latency(
            generate_fn, rows, eval_cfg["latency"]
        )
    else:
        metrics["latency"] = unavailable(
            "no live generator supplied (eval-only rerun); latency requires "
            "generating in-process"
        )
    return metrics
