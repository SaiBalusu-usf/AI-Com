"""Runtime re-linearization (ablations A1/A2) must produce EXACTLY what a
dataset rebuild with that linearization config would produce — the phase-7
review caught v1 reconstructing context from post-cleaning survivors and
prefixing wrong previous-ball outcomes in 37% of rows."""

from cricket_commentary.data.dataset import build_rows, relinearize_rows
from cricket_commentary.data.synthetic import simulate_matches
from cricket_commentary.utils.config import load_config

import pytest


@pytest.fixture(scope="module")
def data_cfg():
    return load_config("configs/data.yaml")


@pytest.fixture(scope="module")
def built(data_cfg):
    matches = simulate_matches(int(data_cfg["seed"]), 3, 20)
    rows, _ = build_rows(matches, data_cfg)
    return matches, rows


def _rebuild_with(data_cfg, matches, lin_cfg):
    cfg = dict(data_cfg, linearization=dict(data_cfg["linearization"], **lin_cfg))
    rows, _ = build_rows(matches, cfg)
    return rows


def test_context_relinearization_matches_full_rebuild(data_cfg, built):
    matches, flat_rows = built
    lin = {"format": "context", "context_balls": 2, "include_game_state": True}
    rebuilt = _rebuild_with(data_cfg, matches, lin)
    runtime = relinearize_rows(flat_rows, lin)
    assert [r["linearized_input"] for r in runtime] == [
        r["linearized_input"] for r in rebuilt
    ]


def test_nogamestate_relinearization_matches_full_rebuild(data_cfg, built):
    matches, flat_rows = built
    lin = {"format": "flat", "include_game_state": False}
    rebuilt = _rebuild_with(data_cfg, matches, lin)
    runtime = relinearize_rows(flat_rows, lin)
    assert [r["linearized_input"] for r in runtime] == [
        r["linearized_input"] for r in rebuilt
    ]


def test_flat_relinearization_is_identity(built):
    _, flat_rows = built
    runtime = relinearize_rows(
        flat_rows, {"format": "flat", "include_game_state": True}
    )
    assert [r["linearized_input"] for r in runtime] == [
        r["linearized_input"] for r in flat_rows
    ]


def test_context_requires_stored_prev_outcomes(built):
    _, flat_rows = built
    stripped = [{k: v for k, v in row.items() if k != "prev_outcomes"} for row in flat_rows[:3]]
    with pytest.raises(ValueError, match="prev_outcomes"):
        relinearize_rows(stripped, {"format": "context"})
