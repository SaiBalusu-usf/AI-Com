import pytest

from cricket_commentary.eval.diversity import distinct_n, diversity_metrics


def _cfg(**overrides):
    cfg = {"distinct_n": [1, 2], "self_bleu_sample_pairs": 200, "repetition_ngram": 4}
    cfg.update(overrides)
    return cfg


def _row(match_id="m1", innings=1, over=1, ball=1):
    return {
        "record": {
            "match_id": match_id,
            "innings": innings,
            "over": over,
            "ball": ball,
            "batter": "A Batter",
            "bowler": "B Bowler",
        },
        "features": {"phase": "powerplay"},
        "linearized_input": "over=0.1 | outcome=DOT",
        "target_commentary": "reference text",
        "split": "test",
    }


def _innings_rows(n, match_id="m1", innings=1):
    return [_row(match_id, innings, over=i // 6 + 1, ball=i % 6 + 1) for i in range(n)]


def test_distinct_n_hand_checked():
    # pooled unigrams: a b a b c -> 3 unique / 5 total
    assert distinct_n(["a b a", "b c"], 1) == pytest.approx(3 / 5)
    # pooled bigrams: (a,b) (b,a) (b,c) -> all unique
    assert distinct_n(["a b a", "b c"], 2) == pytest.approx(1.0)
    # (a,a) twice -> 1 unique / 2 total
    assert distinct_n(["a a a"], 2) == pytest.approx(0.5)
    # lowercased tokens: "The" and "the" are one type
    assert distinct_n(["The the"], 1) == pytest.approx(0.5)


def test_distinct_n_none_when_no_ngrams():
    assert distinct_n([], 1) is None
    assert distinct_n([""], 1) is None
    assert distinct_n(["a"], 2) is None  # one token, no bigrams


def test_identical_generations():
    rows = _innings_rows(4)
    gens = ["the bowler runs in and delivers a sharp yorker"] * 4
    m = diversity_metrics(rows, gens, _cfg(), seed=13)
    # 9 unique unigrams pooled over 36 tokens
    assert m["distinct_1"] == pytest.approx(9 / 36)
    assert m["distinct_1"] < 0.3
    assert m["self_bleu"] == pytest.approx(1.0, abs=1e-6)  # BLEU 100/100
    assert m["repetition_4gram_rate"] == 1.0
    assert m["n_generations"] == 4


def test_all_distinct_generations():
    rows = _innings_rows(4)
    gens = [
        "alpha bravo charlie delta echo",
        "one two three four five six",
        "crimson meadow whispers beneath twilight",
        "quantum ferrets juggle marmalade sideways",
    ]
    m = diversity_metrics(rows, gens, _cfg(), seed=13)
    assert m["repetition_4gram_rate"] == 0.0
    assert m["self_bleu"] < 0.3


def test_repetition_counts_shared_4gram_between_consecutive_balls():
    rows = _innings_rows(3)
    gens = [
        "the ball races away to the boundary",
        "races away to the boundary again fine shot",  # shares "races away to the"
        "completely different words here now",  # shares nothing
    ]
    m = diversity_metrics(rows, gens, _cfg(), seed=13)
    assert m["repetition_4gram_rate"] == pytest.approx(1 / 2)


def test_distinct_is_mean_over_innings_not_pooled():
    rows = _innings_rows(2, innings=1) + _innings_rows(1, innings=2)
    gens = ["a b", "a b", "c d"]
    m = diversity_metrics(rows, gens, _cfg(), seed=13)
    # innings 1: 2 unique / 4 tokens = 0.5; innings 2: 1.0 -> mean 0.75
    # (pooling all texts would give 4/6 instead)
    assert m["distinct_1"] == pytest.approx(0.75)
    assert m["distinct_2"] == pytest.approx(0.75)


def test_none_innings_excluded_from_distinct_mean():
    rows = [_row(innings=1), _row(innings=2)]
    gens = ["word", "a b"]  # innings 1 has no bigrams -> excluded for n=2
    m = diversity_metrics(rows, gens, _cfg(), seed=13)
    assert m["distinct_1"] == pytest.approx(1.0)
    assert m["distinct_2"] == pytest.approx(1.0)


def test_single_generation_innings_pair_metrics_none():
    rows = [_row("m1", 1), _row("m1", 2), _row("m2", 1)]
    gens = ["one lone ball here", "another solitary delivery", "a third one"]
    m = diversity_metrics(rows, gens, _cfg(), seed=13)
    assert m["self_bleu"] is None
    assert m["repetition_4gram_rate"] is None
    assert m["distinct_1"] is not None
    assert m["n_generations"] == 3


def test_all_metrics_none_when_undefined():
    m = diversity_metrics([_row()], [""], _cfg(), seed=13)
    assert m["distinct_1"] is None
    assert m["distinct_2"] is None
    assert m["self_bleu"] is None
    assert m["repetition_4gram_rate"] is None
    assert m["n_generations"] == 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        diversity_metrics([_row(), _row(ball=2)], ["only one"], _cfg(), seed=13)


def _many_pair_case():
    rows = _innings_rows(8)
    gens = [
        f"ball number {i} brings {'a single' if i % 2 else 'no run'} off the bat"
        for i in range(8)
    ]
    return rows, gens


def test_sampling_cap_respected_no_crash():
    rows, gens = _many_pair_case()
    # 8 generations in one innings -> C(8,2) = 28 candidate pairs > cap of 5
    m = diversity_metrics(rows, gens, _cfg(self_bleu_sample_pairs=5), seed=13)
    assert m["self_bleu"] is not None
    assert 0.0 <= m["self_bleu"] <= 1.0


def test_determinism():
    rows, gens = _many_pair_case()
    cfg = _cfg(self_bleu_sample_pairs=5)  # sampling actually kicks in
    a = diversity_metrics(rows, gens, cfg, seed=13)
    b = diversity_metrics(rows, gens, cfg, seed=13)
    assert a == b
