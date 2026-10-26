"""Train, save, reload and infer with real tiny models, entirely offline.

Random models validate API compatibility, never commentary quality. No public
weights, model credentials or paid inference are used by this suite.
"""
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.needs_torch, pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("torch", "transformers", "peft", "trl")),
    reason="requires the complete ML stack",
)]


@pytest.fixture(scope="module")
def local_models(tmp_path_factory):
    import torch
    from cricket_commentary.utils.io import read_jsonl

    spec = importlib.util.spec_from_file_location("tiny_factory", "scripts/make_tiny_model.py")
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    torch.manual_seed(13)
    torch.set_num_threads(2)
    rows = read_jsonl("data/fixtures/mini.jsonl")[:8]
    tokenizer = factory.build_tokenizer([r["linearized_input"] for r in rows])
    folder = tmp_path_factory.mktemp("local-models")
    factory.build_t5(tokenizer, folder / "t5")
    factory.build_causal(tokenizer, folder / "causal")
    return folder, rows


@pytest.mark.parametrize("family", ["t5", "causal"])
def test_training_adapter_reload_and_generation(local_models, tmp_path, monkeypatch, family):
    from cricket_commentary.models.finetune import build_inference, train
    from cricket_commentary.utils.config import load_config

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    folder, rows = local_models
    cfg = load_config(f"configs/smoke/tiny_{family}.yaml")
    cfg["model"]["base"] = str(folder / family)
    cfg["train"]["eval_every_steps"] = 1
    cfg["train"]["batch_size"] = 2
    summary = train(cfg, rows, rows[:2], tmp_path, max_steps=2)
    assert summary["steps"] == 2
    assert Path(summary["adapter_dir"], "adapter_model.safetensors").is_file()
    cfg["model"]["adapter_dir"] = summary["adapter_dir"]
    cfg["decoding"]["max_new_tokens"] = 4
    generator = build_inference(cfg)
    assert isinstance(generator.generate(rows[0]), str)


def test_fewshot_chat_generation(local_models, monkeypatch):
    from cricket_commentary.models.fewshot import FewShotGenerator

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    folder, rows = local_models
    generator = FewShotGenerator({
        "backend": "hf_local", "n_shots": 1, "model_name": str(folder / "causal"),
        "system_prompt": "Describe the ball.", "max_new_tokens": 4,
        "prompt_template": "{examples}\nInput: {input}\nCommentary:",
    }, seed=13, train_rows=rows)
    assert isinstance(generator.generate(rows[0]), str)
