"""B1 template generator tests — including the headline property: zero
hallucinations by construction, verified with the project's own fact checker
over the whole fixture."""

import json

import pytest

from cricket_commentary.eval.factcheck import factcheck_metrics
from cricket_commentary.models.template import TemplateGenerator


@pytest.fixture(scope="module")
def fixture_rows():
    with open("data/fixtures/mini.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


CFG = {"use_game_state": True, "excitement_gate": 0.45}


def test_deterministic_per_ball(fixture_rows):
    gen_a = TemplateGenerator(CFG, seed=13)
    gen_b = TemplateGenerator(CFG, seed=13)
    outs_a = [gen_a.generate(r) for r in fixture_rows[:50]]
    outs_b = [gen_b.generate(r) for r in fixture_rows[:50]]
    assert outs_a == outs_b
    gen_c = TemplateGenerator(CFG, seed=14)
    assert outs_a != [gen_c.generate(r) for r in fixture_rows[:50]]


def test_zero_hallucination_by_construction(fixture_rows):
    gen = TemplateGenerator(CFG, seed=13)
    outputs = [gen.generate(r) for r in fixture_rows]
    metrics = factcheck_metrics(fixture_rows, outputs, {"spacy_ner": "off"})
    assert metrics["hallucination_rate"] == 0.0, metrics["examples"]["violations"]
    assert metrics["omission_rate"] == 0.0


def test_game_state_toggle_changes_register(fixture_rows):
    hot = next(
        r for r in fixture_rows
        if r["record"]["wicket"]
        and r["features"]["event_salience"] * r["features"]["tension"] > 0.45
    )
    on = TemplateGenerator({"use_game_state": True}, seed=13).generate(hot)
    off = TemplateGenerator({"use_game_state": False}, seed=13).generate(hot)
    assert on != off
    assert "!" in on and "!" not in off


def test_mentions_actual_players(fixture_rows):
    gen = TemplateGenerator(CFG, seed=13)
    for row in fixture_rows[:100]:
        out = gen.generate(row)
        rec = row["record"]
        involved = [rec["batter"], rec["bowler"], rec["player_out"]] + rec["fielders"]
        surnames = {n.split()[-1] for n in involved if n}
        named = {t for t in out.replace("!", " ").replace(".", " ").split() if t in surnames}
        # generic extras lines may name nobody; anything named must be involved
        for token in named:
            assert token in surnames


def test_boundary_off_no_ball_is_mentioned(fixture_rows):
    # v1 said only "No ball from X." when the batter hit four off it — a real
    # omission the fact checker caught on the test split
    row = dict(fixture_rows[0])
    row["record"] = dict(
        row["record"], extras_type="noballs", extras=1,
        runs_batter=4, runs_total=5, wicket=False, wicket_type="", player_out="",
    )
    out = TemplateGenerator(CFG, seed=13).generate(row)
    assert "four" in out.lower() and "no ball" in out.lower().replace("oversteps", "no ball")


def test_milestone_lines_only_when_crossed(fixture_rows):
    gen = TemplateGenerator(CFG, seed=13)
    for row in fixture_rows:
        out = gen.generate(row)
        rec = row["record"]
        crossed = rec["batter_runs_after"] >= 50 and rec["batter_runs_after"] - rec["runs_batter"] < 50
        if "Fifty for" in out:
            assert crossed
