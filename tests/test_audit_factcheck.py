"""Checker-accuracy annotation kit: scoring math and the sampling protocol."""

import csv
import json
import subprocess
import sys

import pytest

SCRIPT = "scripts/audit_factcheck.py"
PY = sys.executable


def _write_sheet(path, rows):
    fields = ["match_id", "innings", "over", "delivery_seq", "linearized_input",
              "generation", "checker_hallucination", "checker_omission",
              "checker_detail", "stratum", "human_hallucination",
              "human_omission", "human_notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: "" for k in fields}, **row})


def _row(ch, co, hh, ho, stratum="clean"):
    return {"checker_hallucination": ch, "checker_omission": co,
            "human_hallucination": hh, "human_omission": ho, "stratum": stratum}


def test_scoring_math(tmp_path):
    sheet = tmp_path / "a.csv"
    # hallucination: pred y/gold y (tp), y/n (fp), n/y (fn), n/n x2 (tn)
    _write_sheet(sheet, [
        _row("y", "n", "y", "n", "flagged_hallucination"),
        _row("y", "n", "n", "n", "flagged_hallucination"),
        _row("n", "y", "y", "y", "flagged_omission"),
        _row("n", "n", "n", "n"),
        _row("n", "n", "n", "n"),
    ])
    proc = subprocess.run([PY, SCRIPT, "--score", str(sheet)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    scores = json.loads((tmp_path / "a.scores.json").read_text())
    h = scores["hallucination"]
    assert h["tp"] == 1 and h["fp"] == 1 and h["fn"] == 1 and h["tn"] == 2
    assert h["precision"] == 0.5 and h["recall"] == 0.5
    assert h["accuracy"] == pytest.approx(0.6)
    assert "by_stratum" in h and h["by_stratum"]["flagged_hallucination"]["n"] == 2
    o = scores["omission"]
    assert o["tp"] == 1 and o["fp"] == 0 and o["fn"] == 0


def test_scoring_refuses_unlabelled(tmp_path):
    sheet = tmp_path / "b.csv"
    _write_sheet(sheet, [_row("y", "n", "", "n")])
    proc = subprocess.run([PY, SCRIPT, "--score", str(sheet)],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "finish annotating" in proc.stderr


def test_sampling_is_stratified_and_deterministic(tmp_path):
    import glob

    run_dir = sorted(glob.glob("results/runs/*smoke_tiny_fewshot*"))[-1]
    out_a, out_b = tmp_path / "s1.csv", tmp_path / "s2.csv"
    for out in (out_a, out_b):
        proc = subprocess.run(
            [PY, SCRIPT, "--sample", "30", "--run", run_dir, "--out", str(out)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    assert out_a.read_text() == out_b.read_text()  # seeded
    with open(out_a, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    strata = {r["stratum"] for r in rows}
    # the parroting smoke run has both flagged kinds available
    assert "flagged_hallucination" in strata and "flagged_omission" in strata
    assert all(r["human_hallucination"] == "" for r in rows)
