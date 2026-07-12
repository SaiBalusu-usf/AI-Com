import json
import subprocess

from cricket_commentary.eval.report import git_commit, load_all_runs, write_run_report

RUN_META = {
    "name": "baseline-v1",
    "config_path": "configs/model.yaml",
    "seed": 13,
    "timestamp": "2026-07-12T10:00:00",
}


def test_round_trip(tmp_path):
    metrics = {
        "surface": {"bleu": 0.12345, "chrf": 0.5},
        "judge": {"status": "unavailable", "reason": "no API key configured"},
    }
    report_path = write_run_report(tmp_path / "run1", RUN_META, metrics)

    payload = json.loads((tmp_path / "run1" / "metrics.json").read_text())
    assert payload["run"] == RUN_META
    assert payload["metrics"] == metrics

    assert report_path == tmp_path / "run1" / "report.md"
    text = report_path.read_text()
    assert "baseline-v1" in text
    assert "## surface" in text
    assert "0.1235" in text  # 0.12345 rounded to 4 decimals
    gap_section = text.split("## Metrics not computed")[1]
    assert "judge" in gap_section
    assert "no API key configured" in gap_section


def test_nested_gap_surfaces_with_dotted_path(tmp_path):
    metrics = {
        "quality": {
            "surface": {"bleu": 1.0},
            "semantic": {"status": "unavailable", "reason": "model weights missing"},
        },
        "judge": {"status": "disabled"},
    }
    text = write_run_report(tmp_path, RUN_META, metrics).read_text()
    gap_section = text.split("## Metrics not computed")[1]
    assert "quality.semantic" in gap_section
    assert "model weights missing" in gap_section
    assert "judge" in gap_section
    assert "disabled" in gap_section


def test_no_gaps_renders_none_sentence(tmp_path):
    text = write_run_report(tmp_path, RUN_META, {"surface": {"bleu": 0.5}}).read_text()
    gap_section = text.split("## Metrics not computed")[1]
    assert "None - all configured metrics were computed." in gap_section


def test_load_all_runs_skips_dirs_without_metrics(tmp_path):
    for name in ["b_run", "a_run"]:
        write_run_report(tmp_path / name, {**RUN_META, "name": name}, {"m": {"x": 1.0}})
    (tmp_path / "c_empty").mkdir()  # no metrics.json -> silently skipped

    runs = load_all_runs(tmp_path)
    assert len(runs) == 2
    assert [r["run"]["name"] for r in runs] == ["a_run", "b_run"]  # sorted by dir name
    assert runs[0]["run_dir"] == str(tmp_path / "a_run")
    assert runs[0]["metrics"] == {"m": {"x": 1.0}}

    assert load_all_runs(tmp_path / "does_not_exist") == []


def test_git_commit_none_when_git_missing(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", raise_missing)
    assert git_commit(".") is None
