"""B2 few-shot generator tests: prompt building and example selection are
fully testable without any model; backend loading is exercised only for its
failure contracts in this ML-stack-free environment."""

import importlib.util
import json

import pytest

from cricket_commentary.models.fewshot import (
    FewShotGenerator,
    GeneratorUnavailable,
    select_examples,
)

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

CFG = {
    "backend": "hf_local",
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "n_shots": 8,
    "max_new_tokens": 60,
    "system_prompt": "You are a live cricket commentator.",
    "prompt_template": "{examples}\nInput: {input}\nCommentary:",
}


@pytest.fixture(scope="module")
def rows():
    with open("data/fixtures/mini.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_example_selection_is_stratified_and_seeded(rows):
    a = select_examples(rows, 8, seed=13)
    b = select_examples(rows, 8, seed=13)
    assert [r["linearized_input"] for r in a] == [r["linearized_input"] for r in b]
    classes = set()
    for row in a:
        rec = row["record"]
        if rec["wicket"]:
            classes.add("wicket")
        elif rec["runs_batter"] == 6:
            classes.add("six")
        elif rec["runs_batter"] == 4:
            classes.add("four")
    assert {"wicket", "six", "four"} <= classes  # coverage before top-up
    assert select_examples(rows, 8, seed=14) != a
    with pytest.raises(ValueError):
        select_examples([], 8, seed=13)


@pytest.mark.skipif(HAS_TRANSFORMERS, reason="ML stack present; failure contract n/a")
def test_hf_local_unavailable_in_this_environment(rows):
    with pytest.raises(GeneratorUnavailable, match="setup-full"):
        FewShotGenerator(CFG, seed=13, train_rows=rows)


def test_paid_api_requires_explicit_spend_approval(rows):
    cfg = dict(CFG, backend="anthropic")
    with pytest.raises(GeneratorUnavailable, match="confirm-spend"):
        FewShotGenerator(cfg, seed=13, train_rows=rows, allow_paid_api=False)


def test_prompt_contains_examples_and_input(rows, monkeypatch):
    # bypass backend loading to test prompt assembly in isolation
    monkeypatch.setattr(FewShotGenerator, "_load_hf", lambda self, cfg: ("stub", None, None))
    gen = FewShotGenerator(CFG, seed=13, train_rows=rows)
    prompt = gen.build_prompt(rows[0])
    assert prompt.count("Input:") == 9  # 8 examples + the query
    assert prompt.rstrip().endswith("Commentary:")
    assert rows[0]["linearized_input"] in prompt
    for ex in gen.examples:
        assert ex["target_commentary"] in prompt
