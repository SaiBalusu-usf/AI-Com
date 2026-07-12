"""Surface-overlap metrics (§5.3): sacreBLEU, ROUGE-L, BERTScore.

BLEU and ROUGE-L rely on hard dependencies (sacrebleu, rouge-score) and are
always computable. BERTScore needs bert-score/torch, which may be absent;
availability follows the policy in :mod:`cricket_commentary.eval.base`
(§2 rule 7) — never silently skipped and never fabricated.
"""

from __future__ import annotations

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

from .base import MetricUnavailable, unavailable


def _validate(references: list[str], generations: list[str]) -> None:
    if not references or not generations:
        raise ValueError("surface_metrics: empty input lists")
    if len(references) != len(generations):
        raise ValueError(
            "surface_metrics: length mismatch "
            f"({len(references)} references vs {len(generations)} generations)"
        )


def _bleu(references: list[str], generations: list[str]) -> float:
    """Corpus sacreBLEU on the 0-100 scale, single reference per hypothesis."""
    return float(corpus_bleu(generations, [references]).score)


def _rouge_l(references: list[str], generations: list[str]) -> float:
    """Mean ROUGE-L F-measure over aligned (reference, generation) pairs (0-1)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fmeasures = [
        scorer.score(ref, gen)["rougeL"].fmeasure
        for ref, gen in zip(references, generations)
    ]
    return float(sum(fmeasures) / len(fmeasures))


def _bert_score(references: list[str], generations: list[str], bs_cfg: dict) -> dict:
    """BERTScore per configured mode: auto | off | require (§5.3).

    ``off`` records a deliberate skip; ``auto`` degrades to
    ``{"status": "unavailable", ...}`` when bert-score/torch (or the model
    weights) are missing; ``require`` turns that gap into a hard
    :class:`MetricUnavailable` error.
    """
    mode = bs_cfg["mode"]
    if mode not in ("auto", "off", "require"):
        raise ValueError(f"surface.bert_score.mode must be auto|off|require, got {mode!r}")
    if mode == "off":
        return {"status": "disabled"}
    try:
        import bert_score  # heavy optional dependency (pulls in torch)
    except (ImportError, OSError) as exc:
        # OSError covers torch's broken-install failure mode (missing shared
        # libraries raise at import time, not ImportError)
        reason = f"bert-score/torch not installed or not loadable: {exc}"
        if mode == "require":
            raise MetricUnavailable(reason) from exc
        return unavailable(reason)
    try:
        precision, recall, f1 = bert_score.score(
            cands=generations,
            refs=references,
            model_type=bs_cfg["model_type"],
            rescale_with_baseline=bs_cfg["rescale_with_baseline"],
            lang="en",
        )
    except (ImportError, OSError, MetricUnavailable) as exc:
        reason = f"bert-score could not run ({bs_cfg['model_type']} weights or torch backend missing): {exc}"
        if mode == "require":
            raise MetricUnavailable(reason) from exc
        return unavailable(reason)
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }


def surface_metrics(references: list[str], generations: list[str], cfg: dict) -> dict:
    """Surface-overlap suite over aligned reference/generation lists.

    ``cfg`` is the ``surface`` section of configs/eval.yaml. Returns a dict
    with keys ``bleu`` (0-100), ``rouge_l`` (0-1), and ``bert_score`` (nested
    status/values per availability policy).
    """
    _validate(references, generations)
    return {
        "bleu": _bleu(references, generations) if cfg["bleu"] else {"status": "disabled"},
        "rouge_l": _rouge_l(references, generations) if cfg["rouge_l"] else {"status": "disabled"},
        "bert_score": _bert_score(references, generations, cfg["bert_score"]),
    }
