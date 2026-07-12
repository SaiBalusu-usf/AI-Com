import importlib.util

import pytest

from cricket_commentary.eval.base import MetricUnavailable
from cricket_commentary.eval.excitement import (
    INTENSIFIERS,
    calibration,
    excitement_scores,
    heuristic_excitement,
    validate_on_references,
)

TRANSFORMERS_INSTALLED = importlib.util.find_spec("transformers") is not None

CFG = {
    "mode": "auto",
    "model_name": "j-hartmann/emotion-english-distilroberta-base",
    "model_weight": 0.5,
    "heuristic_weight": 0.5,
    "high_arousal_labels": ["anger", "fear", "joy", "surprise"],
}

EXCITED = [
    "GONE! What a delivery, absolutely brilliant!",
    "SIX! Launches it miles into the stands, huge hit!",
    "What a catch! Stunning, absolutely sensational!",
    "MASSIVE! That is out of here, unbelievable striking!",
    "Crashes it through cover, magnificent shot!",
    "Superb yorker, GONE! The stumps are destroyed!",
]
FLAT = [
    "No run, defended off the back foot.",
    "Single worked to mid on.",
    "Pushed to cover, no run.",
    "Leg bye, they cross for one.",
    "Dot ball, tidy line and length.",
    "One run, nudged to square leg.",
]
HIGH_SIGNAL = [(1.0, 0.9), (0.8, 0.9), (1.0, 0.7), (0.9, 0.85), (0.8, 0.8), (1.0, 0.95)]
LOW_SIGNAL = [(0.0, 0.2), (0.1, 0.25), (0.0, 0.3), (0.1, 0.2), (0.05, 0.3), (0.2, 0.25)]


def _cfg(**overrides):
    cfg = dict(CFG)
    cfg.update(overrides)
    return cfg


def _row(salience, tension, phase, text):
    return {
        "features": {"event_salience": salience, "tension": tension, "phase": phase},
        "target_commentary": text,
    }


def _rows(high_texts, low_texts):
    """12 rows: 6 high-salience*tension death balls + 6 quiet powerplay balls."""
    rows = [
        _row(s, t, "death", text)
        for (s, t), text in zip(HIGH_SIGNAL, high_texts)
    ]
    rows += [
        _row(s, t, "powerplay", text)
        for (s, t), text in zip(LOW_SIGNAL, low_texts)
    ]
    return rows


def test_lexicon_shape():
    assert len(INTENSIFIERS) >= 25
    assert all(phrase == phrase.lower() for phrase in INTENSIFIERS)
    assert any(" " in phrase for phrase in INTENSIFIERS)


def test_exclamation_component():
    assert heuristic_excitement("No run.") == 0.0
    assert heuristic_excitement("No run!") == pytest.approx(0.4 * 1 / 3)
    assert heuristic_excitement("No run!!") == pytest.approx(0.4 * 2 / 3)
    assert heuristic_excitement("No run!!!!") == pytest.approx(0.4)  # capped at 3


def test_intensifier_component():
    assert heuristic_excitement("that is massive") == pytest.approx(0.4 * 1 / 3)
    assert heuristic_excitement("Massive blow") == pytest.approx(0.4 * 1 / 3)  # case-insensitive
    assert heuristic_excitement("what a shot") == pytest.approx(0.4 * 1 / 3)  # multiword phrase
    assert heuristic_excitement("massively built pavilion") == 0.0  # word boundary
    four = "absolutely massive, brilliant, stunning stuff"
    assert heuristic_excitement(four) == pytest.approx(0.4)  # capped at 3


def test_caps_component():
    assert heuristic_excitement("that is SIX over long on") == pytest.approx(0.2 * 1 / 2)
    assert heuristic_excitement("R Khan gets forward") == 0.0  # single-letter initial
    assert heuristic_excitement("SIX SIX SIX") == pytest.approx(0.2)  # capped at 2


def test_bounds():
    maxed = "WHAT A SIX! ABSOLUTELY MASSIVE! GONE! UNBELIEVABLE!"
    assert heuristic_excitement(maxed) == pytest.approx(1.0)
    for text in [maxed, "", "!!!", "R Khan.", *EXCITED, *FLAT]:
        assert 0.0 <= heuristic_excitement(text) <= 1.0


def test_excited_scores_higher_than_flat():
    excited = "What a strike! That is absolutely massive, GONE!"
    flat = "No run, defended solidly."
    assert heuristic_excitement(excited) > heuristic_excitement(flat)


def test_deterministic():
    texts = EXCITED + FLAT
    first, meta_first = excitement_scores(texts, _cfg(mode="heuristic"))
    second, meta_second = excitement_scores(texts, _cfg(mode="heuristic"))
    assert first == second
    assert meta_first == meta_second
    assert heuristic_excitement(EXCITED[0]) == heuristic_excitement(EXCITED[0])


def test_mode_heuristic():
    texts = ["No run.", "SIX! Absolutely massive!"]
    scores, meta = excitement_scores(texts, _cfg(mode="heuristic"))
    assert scores == [heuristic_excitement(text) for text in texts]
    assert meta == {"mode": "heuristic", "fallback_reason": None}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        excitement_scores(["No run."], _cfg(mode="vibes"))


@pytest.mark.skipif(
    TRANSFORMERS_INSTALLED,
    reason="transformers installed; forced model mode would load the real model",
)
def test_forced_model_mode_raises_without_transformers():
    with pytest.raises(MetricUnavailable, match="transformers"):
        excitement_scores(["No run."], _cfg(mode="model+heuristic"))


@pytest.mark.skipif(
    TRANSFORMERS_INSTALLED,
    reason="transformers installed; auto mode may not need to fall back",
)
def test_auto_falls_back_to_heuristic():
    texts = ["No run.", "SIX! Absolutely massive!"]
    scores, meta = excitement_scores(texts, _cfg(mode="auto"))
    assert meta["mode"] == "heuristic"
    assert meta["fallback_reason"] is not None
    assert "transformers" in meta["fallback_reason"]
    assert scores == [heuristic_excitement(text) for text in texts]


def test_validate_on_references_passes_on_aligned_rows():
    rows = _rows(high_texts=EXCITED, low_texts=FLAT)
    report = validate_on_references(rows, _cfg(mode="heuristic"))
    assert report["n"] == 12
    assert report["mode"] == "heuristic"
    assert report["fallback_reason"] is None
    assert report["spearman_rho"] is not None
    assert report["spearman_rho"] > 0
    assert report["passed"] is True
    assert set(report["per_phase"]) == {"death", "powerplay"}
    assert report["per_phase"]["powerplay"] is None  # flat refs all tie: undefined, not 0.0
    assert report["per_phase"]["death"] is not None


def test_validate_on_references_negative_control():
    rows = _rows(high_texts=FLAT, low_texts=EXCITED)  # anti-correlated references
    report = validate_on_references(rows, _cfg(mode="heuristic"))
    assert report["spearman_rho"] is not None
    assert report["spearman_rho"] < 0
    assert report["passed"] is False


def test_calibration_matches_validate_shape():
    rows = _rows(high_texts=EXCITED, low_texts=FLAT)
    generations = [row["target_commentary"] for row in rows]
    report = calibration(rows, generations, _cfg(mode="heuristic"))
    reference_report = validate_on_references(rows, _cfg(mode="heuristic"))
    assert report == reference_report


def test_calibration_length_mismatch_raises():
    rows = _rows(high_texts=EXCITED, low_texts=FLAT)
    with pytest.raises(ValueError):
        calibration(rows, ["only one generation"], _cfg(mode="heuristic"))
