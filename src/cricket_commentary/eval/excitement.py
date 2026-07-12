"""Excitement scoring for commentary text (§5.2).

The v1 excitement score in [0, 1] is a weighted mix of two halves: a model
half (probability mass an emotion classifier puts on high-arousal labels) and
a lexical heuristic half (exclamations, an intensifier lexicon, shouted
ALL-CAPS tokens). Excitement should track the match state — a last-over
wicket reads nothing like a powerplay leave — so before the scorer is trusted
on generations it must pass the mandatory pre-flight
:func:`validate_on_references`: gold references must rank in line with
``event_salience * tension`` (Spearman rho > 0).

Availability policy (§2 rule 7): transformers is optional and never imported
at module top. :func:`model_arousal` raises
:class:`~cricket_commentary.eval.base.MetricUnavailable` rather than
fabricating values; mode "auto" falls back to heuristic-only and records the
reason, while forcing "model+heuristic" without the dependency is a hard
error (configs/eval.yaml).
"""

from __future__ import annotations

import re

from .base import MetricUnavailable, spearman

# Lexical markers of commentary excitement (§5.2). Lowercase; multiword
# phrases are matched as whole phrases with word boundaries.
INTENSIFIERS: list[str] = [
    "what a",
    "out of here",
    "into the stands",
    "absolutely",
    "massive",
    "brilliant",
    "stunning",
    "miles",
    "huge",
    "sensational",
    "magnificent",
    "unbelievable",
    "superb",
    "glorious",
    "spectacular",
    "outrageous",
    "ridiculous",
    "monstrous",
    "blistering",
    "thumping",
    "electric",
    "carnage",
    "roars",
    "launches",
    "crashes",
    "smashes",
    "hammered",
    "clobbered",
    "destroyed",
    "explodes",
    "screamer",
    "maximum",
]

# any whitespace run (space, newline wrap, tab) joins multiword phrases
_INTENSIFIER_PATTERNS = [
    re.compile(
        r"\b" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"\b",
        re.IGNORECASE,
    )
    for phrase in INTENSIFIERS
]
_ALPHA_TOKEN = re.compile(r"[A-Za-z]+")
_MODES = ("auto", "heuristic", "model+heuristic")


def heuristic_excitement(text: str) -> float:
    """Lexical excitement proxy in [0, 1] — the heuristic half of §5.2.

    With ``n_exclaim`` = count of ``"!"``, ``n_intens`` = count of
    :data:`INTENSIFIERS` hits (case-insensitive, word/phrase boundaries), and
    ``n_caps`` = count of alphabetic tokens of >= 3 characters written fully
    in uppercase ("SIX" and "GONE" count; the initial in "R Khan" does not):

        score = 0.4 * min(n_exclaim, 3) / 3
              + 0.4 * min(n_intens, 3) / 3
              + 0.2 * min(n_caps, 2) / 2

    Each component saturates so one screamed word cannot dominate the scale.
    """
    n_exclaim = text.count("!")
    n_intens = sum(len(pattern.findall(text)) for pattern in _INTENSIFIER_PATTERNS)
    n_caps = sum(
        1 for token in _ALPHA_TOKEN.findall(text) if len(token) >= 3 and token.isupper()
    )
    return (
        0.4 * (min(n_exclaim, 3) / 3)
        + 0.4 * (min(n_intens, 3) / 3)
        + 0.2 * (min(n_caps, 2) / 2)
    )


def model_arousal(texts: list[str], cfg: dict) -> list[float]:
    """High-arousal probability per text — the model half of §5.2.

    Runs ``cfg["model_name"]`` through the transformers text-classification
    pipeline with ``top_k=None`` and, per text, sums the probabilities of the
    labels in ``cfg["high_arousal_labels"]``. Raises
    :class:`MetricUnavailable` when transformers is not installed or the
    model cannot be loaded — never a fabricated number (§2 rule 7).
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise MetricUnavailable(
            f"excitement model half needs transformers, which is not installed: {exc}"
        ) from exc
    try:
        classifier = pipeline("text-classification", model=cfg["model_name"], top_k=None)
        outputs = classifier(list(texts))
    except (ImportError, OSError) as exc:
        raise MetricUnavailable(
            f"could not load excitement model {cfg['model_name']!r}: {exc}"
        ) from exc
    high = set(cfg["high_arousal_labels"])
    # clamp: a multi-label (sigmoid) classifier configured via model_name
    # could sum past 1.0, which would leak out of the score's [0,1] contract
    return [
        min(1.0, max(0.0, float(sum(
            item["score"] for item in output if item["label"] in high
        ))))
        for output in outputs
    ]


def excitement_scores(texts: list[str], cfg: dict) -> tuple[list[float], dict]:
    """Excitement in [0, 1] per text, plus meta describing which halves ran.

    ``cfg["mode"]``:

    - ``"heuristic"``: heuristic half only.
    - ``"model+heuristic"``: weighted mix of both halves;
      :class:`MetricUnavailable` propagates (forcing the mode on a machine
      without the dependency is a hard error, configs/eval.yaml).
    - ``"auto"``: model+heuristic when the model is available, otherwise
      heuristic-only with the reason recorded in ``meta["fallback_reason"]``.

    ``cfg["model_weight"]`` / ``cfg["heuristic_weight"]`` are renormalised to
    sum to 1 over the halves that actually ran, so a heuristic-only run
    returns the heuristic score unchanged.

    Returns ``(scores, {"mode": "model+heuristic" | "heuristic",
    "fallback_reason": str | None})``. Unknown modes raise ``ValueError``.
    """
    mode = cfg["mode"]
    if mode not in _MODES:
        raise ValueError(f"unknown excitement mode {mode!r}; expected one of {_MODES}")

    heuristic = [heuristic_excitement(text) for text in texts]
    if mode == "heuristic":
        return heuristic, {"mode": "heuristic", "fallback_reason": None}

    try:
        model = model_arousal(texts, cfg)
    except MetricUnavailable as exc:
        if mode == "model+heuristic":
            raise
        return heuristic, {"mode": "heuristic", "fallback_reason": str(exc)}

    model_weight = float(cfg["model_weight"])
    heuristic_weight = float(cfg["heuristic_weight"])
    if model_weight < 0 or heuristic_weight < 0 or model_weight + heuristic_weight <= 0:
        raise ValueError(
            "excitement weights must be non-negative and sum to > 0, got "
            f"model_weight={model_weight}, heuristic_weight={heuristic_weight}"
        )
    total = model_weight + heuristic_weight
    scores = [
        (model_weight * m + heuristic_weight * h) / total
        for m, h in zip(model, heuristic)
    ]
    return scores, {"mode": "model+heuristic", "fallback_reason": None}


def _correlate(rows: list[dict], texts: list[str], cfg: dict) -> dict:
    """Spearman of excitement scores against event_salience * tension."""
    scores, meta = excitement_scores(texts, cfg)
    signal = [
        float(row["features"]["event_salience"]) * float(row["features"]["tension"])
        for row in rows
    ]
    rho = spearman(scores, signal)

    by_phase: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_phase.setdefault(row["features"]["phase"], []).append(i)
    per_phase = {
        phase: spearman([scores[i] for i in idx], [signal[i] for i in idx])
        for phase, idx in by_phase.items()
    }

    return {
        "spearman_rho": rho,
        "n": len(rows),
        "mode": meta["mode"],
        "fallback_reason": meta["fallback_reason"],
        "per_phase": per_phase,
        "passed": rho is not None and rho > 0,
    }


def validate_on_references(rows: list[dict], cfg: dict) -> dict:
    """Mandatory pre-flight (§5.2): sanity-check the scorer on gold text.

    Scores every ``row["target_commentary"]`` and correlates (Spearman)
    against ``features.event_salience * features.tension``. Returns
    ``{"spearman_rho", "n", "mode", "fallback_reason",
    "per_phase": {phase: rho-or-None}, "passed"}`` where ``passed`` requires
    a defined, strictly positive rho. A scorer that cannot rank references in
    line with the match state must not be trusted on generations; undefined
    correlations stay ``None`` rather than becoming 0.0 (§2 rule 7).
    """
    return _correlate(rows, [row["target_commentary"] for row in rows], cfg)


def calibration(rows: list[dict], generations: list[str], cfg: dict) -> dict:
    """Correlation report for system generations, same shape as
    :func:`validate_on_references` (§5.2).

    ``generations`` must align 1:1 with ``rows``; a length mismatch raises
    ``ValueError``.
    """
    if len(rows) != len(generations):
        raise ValueError(
            f"calibration: {len(rows)} rows but {len(generations)} generations"
        )
    return _correlate(rows, list(generations), cfg)
