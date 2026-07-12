"""Diversity metrics (§5.3): the "EA Sports repetition" battery.

Generated commentary can be fluent yet numbingly repetitive — the video-game
announcer failure mode. This module measures lexical variety (distinct-n),
cross-generation similarity within an innings (self-BLEU), and literal phrase
recycling between consecutive balls (repetition rate). Pair sampling for
self-BLEU is driven by an explicit seed. A metric that is undefined for the
input is reported as None, never fabricated (§2 rule 7).
"""

from __future__ import annotations

import numpy as np
from sacrebleu.metrics import BLEU

from .base import group_by_innings


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float | None:
    """Unique n-grams / total n-grams, pooled over all texts.

    Tokens are lowercased whitespace splits. Returns None when the pooled
    texts yield no n-grams at all (empty input, or every text shorter than
    n tokens) — that is undefined, not zero diversity.
    """
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        grams = _ngrams(_tokens(text), n)
        total += len(grams)
        unique.update(grams)
    if total == 0:
        return None
    return len(unique) / total


def _self_bleu(
    groups: dict[tuple[str, int], list[int]],
    generations: list[str],
    sample_pairs: int,
    seed: int,
) -> float | None:
    """Mean sentence-BLEU/100 over sampled within-innings generation pairs.

    Candidate pairs are all unordered pairs (i < j) of generations from the
    same innings; up to ``sample_pairs`` of them are drawn without replacement
    via ``numpy.random.default_rng(seed)``. None when no innings has >= 2
    generations (no pair exists).
    """
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
    k = min(int(sample_pairs), len(pairs))
    if k <= 0:
        return None
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pairs), size=k, replace=False)
    bleu = BLEU(effective_order=True)
    scores = []
    for c in chosen:
        i, j = pairs[int(c)]
        # sentence-BLEU is asymmetric (brevity penalty); average both
        # directions so the estimate does not depend on ball order
        forward = bleu.sentence_score(generations[i], [generations[j]]).score
        backward = bleu.sentence_score(generations[j], [generations[i]]).score
        scores.append((forward + backward) / 200.0)
    # sacrebleu returns 100.00000000000004 on identical pairs; clamp the mean
    return float(min(max(np.mean(scores), 0.0), 1.0))


def _repetition_rate(
    groups: dict[tuple[str, int], list[int]],
    generations: list[str],
    n: int,
) -> float | None:
    """Fraction of consecutive within-innings generation pairs (ball order as
    given) that recycle phrasing: sharing at least one common n-gram, or being
    the exact same text (catches verbatim outputs shorter than the n-gram
    window, e.g. "no run" repeated). Pairs of *differing* texts where either
    side has fewer than n tokens cannot share an n-gram by construction and
    are excluded from the denominator rather than counted as fresh. None when
    no measurable pair exists."""
    n_pairs = 0
    n_shared = 0
    for indices in groups.values():
        for prev, cur in zip(indices, indices[1:]):
            prev_tokens = _tokens(generations[prev])
            cur_tokens = _tokens(generations[cur])
            if prev_tokens == cur_tokens:
                n_pairs += 1
                n_shared += 1
                continue
            if len(prev_tokens) < n or len(cur_tokens) < n:
                continue  # sharing is impossible; not evidence of freshness
            n_pairs += 1
            if set(_ngrams(prev_tokens, n)) & set(_ngrams(cur_tokens, n)):
                n_shared += 1
    if n_pairs == 0:
        return None
    return n_shared / n_pairs


def diversity_metrics(
    rows: list[dict], generations: list[str], cfg: dict, seed: int
) -> dict:
    """Diversity battery over ``generations`` aligned 1:1 with ``rows``.

    Returns a dict with:

    - ``distinct_<n>`` for each n in ``cfg["distinct_n"]``: mean of
      per-innings :func:`distinct_n` (innings grouped via
      :func:`base.group_by_innings`; innings whose value is None are excluded
      from the mean; None overall when every innings is None).
    - ``self_bleu``: mean sacrebleu sentence-BLEU/100 over up to
      ``cfg["self_bleu_sample_pairs"]`` sampled within-innings pairs.
    - ``repetition_<n>gram_rate`` (n = ``cfg["repetition_ngram"]``): fraction
      of consecutive within-innings pairs recycling phrasing (shared n-gram or
      verbatim-identical text).
    - ``n_generations``: number of generations scored.

    Raises ValueError when rows and generations differ in length.
    """
    if len(rows) != len(generations):
        raise ValueError(
            f"diversity_metrics: {len(rows)} rows but {len(generations)} generations"
        )

    groups = group_by_innings(rows)
    out: dict = {}

    for n in cfg["distinct_n"]:
        values: list[float] = []
        for indices in groups.values():
            value = distinct_n([generations[i] for i in indices], int(n))
            if value is not None:
                values.append(value)
        out[f"distinct_{n}"] = float(np.mean(values)) if values else None

    out["self_bleu"] = _self_bleu(
        groups, generations, int(cfg["self_bleu_sample_pairs"]), seed
    )
    rep_n = int(cfg["repetition_ngram"])
    out[f"repetition_{rep_n}gram_rate"] = _repetition_rate(groups, generations, rep_n)
    out["n_generations"] = len(generations)
    return out
