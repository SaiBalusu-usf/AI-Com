import pytest

from cricket_commentary.utils.config import (
    ConfigError,
    apply_overrides,
    load_config,
    require,
)


def test_all_shipped_configs_load_and_are_seeded(configs_dir):
    for path in sorted(configs_dir.rglob("*.yaml")):
        cfg = load_config(path)
        assert isinstance(cfg.get("seed"), int), f"{path} must define an int seed"


def test_extends_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ntrain: {lr: 0.1, epochs: 3}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain: {lr: 0.2}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["train"] == {"lr": 0.2, "epochs": 3}
    assert cfg["seed"] == 1


def test_overrides_parse_yaml_scalars():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=true", "d=hello"])
    assert cfg["a"]["b"] == 2 and cfg["a"]["c"] is True and cfg["d"] == "hello"


def test_require_names_missing_key():
    with pytest.raises(ConfigError, match="train.lr"):
        require({"_config_path": "x.yaml"}, "train.lr")


def test_missing_config_fails_loudly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
