import pytest

from cricket_commentary.eval.base import MetricUnavailable
from cricket_commentary.eval.surface import surface_metrics

REFS = [
    "The batter drives through the covers for four glorious runs.",
    "A sharp bouncer and the batter ducks underneath it comfortably.",
    "Big appeal for lbw but the umpire shakes his head firmly.",
]
DISJOINT = [
    "purple engines hum quietly beneath frozen midnight glaciers tonight.",
    "seventeen wandering llamas juggled neon umbrellas near basalt cliffs.",
    "quantum spreadsheets whisper obsolete lullabies toward rusty satellites.",
]


def _cfg(mode: str = "off") -> dict:
    return {
        "bleu": True,
        "rouge_l": True,
        "bert_score": {
            "mode": mode,
            "model_type": "roberta-large",
            "rescale_with_baseline": True,
        },
    }


def _bert_score_importable() -> bool:
    try:
        import bert_score  # noqa: F401
    except ImportError:
        return False
    return True


def test_identical_texts_score_near_perfect():
    out = surface_metrics(REFS, list(REFS), _cfg())
    assert out["bleu"] > 99
    assert out["rouge_l"] == pytest.approx(1.0)


def test_disjoint_texts_score_near_zero():
    out = surface_metrics(REFS, DISJOINT, _cfg())
    assert out["bleu"] < 5
    assert out["rouge_l"] < 0.2


def test_bert_score_mode_off_is_disabled():
    out = surface_metrics(REFS, list(REFS), _cfg(mode="off"))
    assert out["bert_score"] == {"status": "disabled"}


@pytest.mark.skipif(
    _bert_score_importable(),
    reason="bert-score is installed; the unavailable path is not reachable here",
)
def test_bert_score_mode_auto_records_unavailable():
    out = surface_metrics(REFS, list(REFS), _cfg(mode="auto"))
    result = out["bert_score"]
    assert result["status"] == "unavailable"
    assert "bert-score" in result["reason"] or "torch" in result["reason"]


@pytest.mark.skipif(
    _bert_score_importable(),
    reason="bert-score is installed; the unavailable path is not reachable here",
)
def test_bert_score_mode_require_raises():
    with pytest.raises(MetricUnavailable):
        surface_metrics(REFS, list(REFS), _cfg(mode="require"))


def test_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        surface_metrics(REFS, REFS[:2], _cfg())


def test_empty_inputs_raise_value_error():
    with pytest.raises(ValueError):
        surface_metrics([], [], _cfg())
    with pytest.raises(ValueError):
        surface_metrics([], REFS, _cfg())


def test_determinism():
    first = surface_metrics(REFS, DISJOINT, _cfg())
    second = surface_metrics(REFS, DISJOINT, _cfg())
    assert first == second
