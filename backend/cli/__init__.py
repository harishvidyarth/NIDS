"""`nids` — one command-line entrypoint over the whole NIDS pipeline.

Usage: `python -m backend.cli <group> <command> [options]` or
`python scripts/nids.py ...`. Every subcommand calls the same service
function as the matching `/api/*` route; see `backend/cli/commands.py`.
"""
from __future__ import annotations

import argparse
import sys

from . import commands
from .format import CliError, emit

_TEMPORAL_WINDOW_DEFAULT = 10
_TEMPORAL_SEQ_DEFAULT = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nids", description="NIDS pipeline CLI")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a text summary")
    sub = parser.add_subparsers(dest="group", required=True)

    # serve
    p = sub.add_parser("serve", help="run the FastAPI app (uvicorn)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=commands.cmd_serve)

    # interfaces
    p = sub.add_parser("interfaces", help="list capture interfaces")
    p.set_defaults(func=commands.cmd_interfaces)

    # capture
    p = sub.add_parser("capture", help="foreground packet capture / pcap inspection")
    cap = p.add_subparsers(dest="capture_cmd", required=True)
    c = cap.add_parser("start", help="capture until --seconds elapses or Ctrl+C")
    c.add_argument("--iface", required=True, help="interface id, or 'all'")
    c.add_argument("--seconds", type=int, default=300)
    c.add_argument("--packets", type=int, default=None)
    c.add_argument("--buffer-mb", dest="buffer_mb", type=int, default=64)
    c = cap.add_parser("packets", help="read packets from a pcap (paginated)")
    c.add_argument("--pcap", required=True)
    c.add_argument("--limit", type=int, default=100)
    c.add_argument("--offset", type=int, default=0)
    c = cap.add_parser("stat", help="validate + capinfos a pcap")
    c.add_argument("--pcap", required=True)
    p.set_defaults(func=commands.cmd_capture)

    # extract
    p = sub.add_parser("extract", help="pcap -> CICFlowMeter feature CSV")
    p.add_argument("pcap")
    p.add_argument("--features-dir", dest="features_dir", default=None)
    p.set_defaults(func=commands.cmd_extract)

    # predict
    p = sub.add_parser("predict", help="per-flow ANN prediction on a feature CSV")
    p.add_argument("csv")
    p.add_argument("--out", default=None, help="write the full result JSON here")
    p.set_defaults(func=commands.cmd_predict)

    # predict-metrics
    p = sub.add_parser("predict-metrics", help="ANN precision/recall/F1/FPR + confusion matrix")
    p.add_argument("--csv-path", dest="csv_path", default=None)
    p.set_defaults(func=commands.cmd_predict_metrics)

    # explain
    p = sub.add_parser("explain", help="SHAP / gradient attribution job for a model input")
    p.add_argument("--model", required=True, choices=("ann", "lstm", "hgb"))
    p.add_argument("values", help="path to a .npy of already-scaled model input")
    p.add_argument("--class", dest="explained_class", default=None)
    p.add_argument("--wait", action="store_true", help="block until the job finishes")
    p.add_argument("--timeout", type=int, default=120)
    p.set_defaults(func=commands.cmd_explain)

    # temporal
    p = sub.add_parser("temporal", help="temporal windowed dataset build / validate")
    tmp = p.add_subparsers(dest="temporal_cmd", required=True)
    t = tmp.add_parser("prepare", help="feature CSV -> windows / transitions / sequences / splits")
    t.add_argument("csv")
    t.add_argument("--window", type=int, default=_TEMPORAL_WINDOW_DEFAULT)
    t.add_argument("--seq", type=int, default=_TEMPORAL_SEQ_DEFAULT)
    t.add_argument("--out", default=None)
    t = tmp.add_parser("validate", help="10-check audit of a prepared temporal dataset")
    t.add_argument("dir")
    t.add_argument("--source-csv", dest="source_csv", required=True)
    p.set_defaults(func=commands.cmd_temporal)

    # lstm (Phase 3)
    p = sub.add_parser("lstm", help="one-step next-window LSTM forecaster")
    lstm = p.add_subparsers(dest="lstm_cmd", required=True)
    lstm.add_parser("status", help="training job status")
    l = lstm.add_parser("train", help="start a training job")
    l.add_argument("--force", action="store_true")
    l = lstm.add_parser("forecast", help="one-window forecast from the active model")
    l.add_argument("--windows", default=None, help="prepared temporal dataset dir")
    l = lstm.add_parser("evaluate", help="rigorous evaluation of a saved artifact")
    l.add_argument("--artifact-dir", dest="artifact_dir", default=None)
    lstm.add_parser("report", help="print the frozen evaluation report JSON")
    p.set_defaults(func=commands.cmd_lstm)

    # multistep (Phase 4)
    p = sub.add_parser("multistep", help="direct H1-H6 multi-step forecaster")
    ms = p.add_subparsers(dest="multistep_cmd", required=True)
    m = ms.add_parser("dataset", help="build the multi-step dataset")
    m.add_argument("--force", action="store_true")
    m = ms.add_parser("train", help="train the multi-step model")
    m.add_argument("--force", action="store_true")
    ms.add_parser("evaluate", help="print the evaluation report JSON")
    ms.add_parser("benchmark", help="print the performance benchmark JSON")
    p.set_defaults(func=commands.cmd_multistep)

    # worldmodel
    p = sub.add_parser("worldmodel", help="K-step autoregressive infiltration forecast")
    wm = p.add_subparsers(dest="worldmodel_cmd", required=True)
    wm.add_parser("status", help="training job status")
    w = wm.add_parser("train", help="start a training job (needs NIDS_CICIDS2017_DIR)")
    w.add_argument("--force", action="store_true")
    w = wm.add_parser("forecast", help="per-step infiltration probability + MITRE stage")
    w.add_argument("--k", type=int, default=None)
    w.add_argument("--windows", default=None, help="prepared temporal dataset dir")
    p.set_defaults(func=commands.cmd_worldmodel)

    # benchmark
    p = sub.add_parser("benchmark", help="LSTM vs logistic-regression baseline (incl. FPR)")
    p.set_defaults(func=commands.cmd_benchmark)

    # mitre
    p = sub.add_parser("mitre", help="map a forecast state to ATT&CK candidates")
    mit = p.add_subparsers(dest="mitre_cmd", required=True)
    m = mit.add_parser("map")
    m.add_argument("--state", required=True, help="current state, e.g. PortScan")
    m.add_argument("--prob", action="append", metavar="CLASS=VALUE", help="repeatable")
    m.add_argument("--feature", action="append", metavar="KEY=VALUE", help="repeatable")
    p.set_defaults(func=commands.cmd_mitre)

    # pipeline
    p = sub.add_parser("pipeline", help="on-disk artifact / report inventory")
    pl = p.add_subparsers(dest="pipeline_cmd", required=True)
    pl.add_parser("show")
    p.set_defaults(func=commands.cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # accept --json anywhere (argparse would otherwise reject it after the
    # subcommand, which is where people naturally type it).
    as_json = any(flag in argv for flag in ("--json", "-j"))
    argv = [a for a in argv if a not in ("--json", "-j")]

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except CliError as error:
        print(f"error: {error.message}", file=sys.stderr)
        return error.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    emit(result, as_json=as_json or getattr(args, "json", False))
    return 0
