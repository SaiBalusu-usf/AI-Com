"""D1 loader tests: Kaggle-style commentary CSVs -> Cricsheet-format matches
+ delivery-keyed reference texts, exercised end-to-end through build_rows.
CSV samples are handcrafted to mimic the two candidate datasets' conventions
(combined 'over_ball' notation vs separate 1-based over columns)."""

import pytest

from cricket_commentary.data.dataset import build_rows
from cricket_commentary.data.kaggle_commentary import (
    ColumnMapError,
    audit_alignment,
    load_commentary_csv,
    resolve_columns,
)
from cricket_commentary.utils.config import load_config

# convention A: combined scoreboard over.ball, batsman/batsman_runs naming.
# Includes a wide followed by a legal ball with DISTINCT texts, and a wicket.
CSV_A = """match_id,innings,over_ball,batting_team,batsman,bowler,non_striker,batsman_runs,extras,extra_type,total_runs,is_wicket,dismissal_kind,player_dismissed,commentary
m1,1,0.1,Team Alpha,V Kohli,T Boult,S Gill,0,0,,0,False,,,"Boult to Kohli, no run, defended into the covers"
m1,1,0.2,Team Alpha,V Kohli,T Boult,S Gill,0,1,wides,1,False,,,"Boult strays down leg, wide called by the umpire"
m1,1,0.2,Team Alpha,V Kohli,T Boult,S Gill,4,0,,4,False,,,"Kohli crunches it through the covers for four"
m1,1,0.3,Team Alpha,V Kohli,T Boult,S Gill,0,0,,0,True,bowled,V Kohli,"Bowled him! Kohli plays all over it and departs"
m1,2,0.1,Team Beta,T Boult,V Kohli,M Sharma,1,0,,1,False,,,"Boult tucks a quick single off Kohli to get going"
"""

# convention B: separate 1-based over column, striker/runs_off_bat naming
CSV_B = """match,inning,over,ball,team,striker,bowler_name,runs_off_bat,extra_runs,kind_of_extras,text
game7,1,1,1,Team Gamma,R Sharma,P Cummins,6,0,,"Sharma launches Cummins over long-on, a huge six"
game7,1,1,2,Team Gamma,R Sharma,P Cummins,1,0,,"Sharma works a single into the leg side"
"""


@pytest.fixture()
def csv_a(tmp_path):
    path = tmp_path / "conv_a.csv"
    path.write_text(CSV_A)
    return path


@pytest.fixture()
def csv_b(tmp_path):
    path = tmp_path / "conv_b.csv"
    path.write_text(CSV_B)
    return path


def test_resolve_columns_defaults_and_overrides():
    header = ["match_id", "innings", "over_ball", "batsman", "bowler",
              "total_runs", "commentary"]
    cols = resolve_columns(header)
    assert cols["batter"] == "batsman" and cols["commentary"] == "commentary"
    cols = resolve_columns(header, {"commentary": "total_runs"})  # explicit wins
    assert cols["commentary"] == "total_runs"
    with pytest.raises(ColumnMapError, match="not in CSV header"):
        resolve_columns(header, {"batter": "nope"})
    with pytest.raises(ColumnMapError, match="cannot locate"):
        resolve_columns(["foo", "bar"])


def test_convention_a_roundtrip(csv_a):
    matches, texts = load_commentary_csv(csv_a)
    assert set(matches) == {"m1"}
    innings = matches["m1"]["innings"]
    assert [i["team"] for i in innings] == ["Team Alpha", "Team Beta"]

    deliveries = innings[0]["overs"][0]["deliveries"]
    assert len(deliveries) == 4
    assert deliveries[1]["extras"] == {"wides": 1}          # typed correctly
    assert deliveries[3]["wickets"][0]["kind"] == "bowled"

    # the wide (seq 2) and the following legal ball (seq 3) share over.ball
    # "0.2" but must carry DIFFERENT texts under the delivery_seq join
    assert "wide called" in texts[("m1", 1, 1, 2)]
    assert "for four" in texts[("m1", 1, 1, 3)]


def test_convention_b_one_based_overs_normalised(csv_b):
    matches, texts = load_commentary_csv(csv_b)
    over_block = matches["game7"]["innings"][0]["overs"][0]
    assert over_block["over"] == 0                          # 1-based -> cricsheet 0-based
    # runs derived from runs_off_bat; totals synthesised
    assert over_block["deliveries"][0]["runs"] == {"batter": 6, "extras": 0, "total": 6}
    assert "huge six" in texts[("game7", 1, 1, 1)]


def test_full_pipeline_uses_real_commentary(csv_a):
    matches, texts = load_commentary_csv(csv_a)
    cfg = dict(load_config("configs/data.yaml"))
    cfg["commentary"] = dict(cfg["commentary"], source="kaggle_test")
    cfg["source"] = dict(cfg["source"], kind="kaggle_commentary")
    rows, summary = build_rows(matches, cfg, reference_texts=texts)
    assert summary["balls_without_commentary"] == 0
    assert all(row["reference_source"] == "kaggle_test" for row in rows)
    wide_row = next(r for r in rows if r["record"]["extras_type"] == "wides")
    assert "wide called" in wide_row["target_commentary"]
    four_row = next(r for r in rows if r["record"]["runs_batter"] == 4)
    assert "for four" in four_row["target_commentary"]
    # cumulative state derived by the audited ingest math, not trusted CSV
    assert four_row["record"]["team_score"] == 5
    audit_alignment(summary)  # must pass on aligned data


def test_alignment_audit_fails_loudly():
    summary = {"cleaning": {"input_rows": 100, "dropped_entity_mismatch": 30}}
    with pytest.raises(ColumnMapError, match="alignment audit FAILED"):
        audit_alignment(summary, max_mismatch_rate=0.2)


def test_missing_runs_columns_fail(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("batsman,bowler,over_ball,commentary\nA,B,0.1,some text here\n")
    with pytest.raises(ColumnMapError, match="runs"):
        load_commentary_csv(path)
