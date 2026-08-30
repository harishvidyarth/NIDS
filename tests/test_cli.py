"""Tests for the `nids` CLI (backend/cli).

These exercise argument parsing, dispatch, output formatting and exit
codes without touching TensorFlow — heavy handlers are monkeypatched.
"""
from __future__ import annotations

import json

import pytest

from backend.cli import build_parser, main
from backend.cli import commands
from backend.cli.format import CliError, emit


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def test_parser_wires_every_group():
    parser = build_parser()
    groups = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert {
        "serve", "interfaces", "capture", "extract", "predict", "predict-metrics",
        "explain", "temporal", "lstm", "multistep", "worldmodel", "benchmark",
        "mitre", "pipeline",
    } <= set(groups)


def test_parser_sets_handler_for_subcommand():
    args = build_parser().parse_args(["worldmodel", "forecast", "--k", "4"])
    assert args.func is commands.cmd_worldmodel
    assert args.worldmodel_cmd == "forecast"
    assert args.k == 4


def test_missing_subcommand_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["lstm"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# format helpers
# --------------------------------------------------------------------------
def test_emit_json(capsys):
    emit({"a": 1, "b": [1, 2]}, as_json=True)
    assert json.loads(capsys.readouterr().out) == {"a": 1, "b": [1, 2]}


def test_emit_human_is_indented(capsys):
    emit({"outer": {"inner": 5}}, as_json=False)
    out = capsys.readouterr().out
    assert "outer:" in out and "  inner: 5" in out


def test_kv_pairs_float_and_error():
    assert commands._kv_pairs(["a=1.5", "b=2"], cast_float=True) == {"a": 1.5, "b": 2.0}
    with pytest.raises(CliError) as exc:
        commands._kv_pairs(["bad"], cast_float=False)
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# dispatch + exit codes
# --------------------------------------------------------------------------
def test_clierror_code_is_returned(monkeypatch, capsys):
    monkeypatch.setattr(commands, "cmd_benchmark", lambda args: (_ for _ in ()).throw(CliError(4, "boom")))
    assert main(["benchmark"]) == 4
    assert "boom" in capsys.readouterr().err


def test_ok_path_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(commands, "cmd_benchmark", lambda args: {"ok": True})
    assert main(["--json", "benchmark"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_pipeline_show_runs_for_real(capsys):
    assert main(["--json", "pipeline", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "worldmodel_trained" in payload
    assert "shap_backgrounds" in payload


def test_json_flag_is_position_agnostic(capsys):
    assert main(["pipeline", "show", "--json"]) == 0
    json.loads(capsys.readouterr().out)  # trailing --json still produces JSON


def test_worldmodel_forecast_without_artifact_exits_3(capsys):
    # no artifacts/worldmodel/latest.json in the tree -> WorldModelUnavailable
    code = main(["worldmodel", "forecast"])
    assert code == 3
    assert "world-model" in capsys.readouterr().err.lower()


def test_mitre_map_runs_for_real(capsys):
    code = main([
        "--json", "mitre", "map", "--state", "PortScan",
        "--prob", "BENIGN=0.1", "--prob", "PortScan=0.9",
        "--feature", "unique_dst_port_count=40", "--feature", "syn_count=55",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["predicted_next_state"] == "PortScan"
    assert any(c["technique_id"] == "T1595" for c in payload["mitre_candidates"])


def test_mitre_map_requires_a_probability(capsys):
    assert main(["mitre", "map", "--state", "BENIGN"]) == 2


@pytest.mark.parametrize("group", ["lstm", "multistep", "worldmodel"])
def test_train_status_forecast_groups_require_a_subcommand(group):
    with pytest.raises(SystemExit) as exc:
        main([group])
    assert exc.value.code == 2
