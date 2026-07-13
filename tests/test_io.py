import pytest

from cricket_commentary.utils.io import read_jsonl, write_jsonl


def test_jsonl_roundtrip_is_utf8_with_lf(tmp_path):
    path = tmp_path / "x.jsonl"
    rows = [{"text": "up and over — six runs"}, {"text": "año nuevo"}]
    write_jsonl(path, rows)
    raw = path.read_bytes()
    assert b"\r\n" not in raw                 # platform-stable line endings
    assert "—".encode("utf-8") in raw          # utf-8, not locale codec
    assert read_jsonl(path) == rows


def test_read_jsonl_reports_legacy_encoding_actionably(tmp_path):
    # a cp1252 em-dash (0x97) — the exact corruption a pre-UTF-8-pin Windows
    # run left behind in regenerated files (student repro, 2026-07-13)
    bad = tmp_path / "legacy.jsonl"
    bad.write_bytes(b'{"text": "up and over \x97 six runs"}\n')
    with pytest.raises(ValueError, match="not UTF-8"):
        read_jsonl(bad)


def test_read_jsonl_reports_bad_json_with_line_number(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(bad)
