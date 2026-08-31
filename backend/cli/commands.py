"""Subcommand handlers for the `nids` CLI.

Every handler is a thin wrapper over an existing service function — the
same one the matching FastAPI route in `backend/api/main.py` calls. No
pipeline logic is reimplemented here. Heavy imports (TensorFlow, the
model modules) are done inside each handler so `nids --help` and unrelated
subcommands stay fast.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .format import CliError


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _repo_root() -> Path:
    from ..config import REPO_ROOT

    return REPO_ROOT


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise CliError(3, f"Not found on disk: {path}")
    return json.loads(path.read_text())


def _kv_pairs(pairs: list[str] | None, *, cast_float: bool) -> dict:
    out: dict[str, object] = {}
    for item in pairs or []:
        if "=" not in item:
            raise CliError(2, f"Expected KEY=VALUE, got: {item!r}")
        key, _, raw = item.partition("=")
        if cast_float:
            try:
                out[key] = float(raw)
            except ValueError as exc:
                raise CliError(2, f"{key}={raw!r} is not a number") from exc
        else:
            out[key] = raw
    return out


# --------------------------------------------------------------------------
# server / interfaces
# --------------------------------------------------------------------------
def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("backend.api.main:app", host=args.host, port=args.port, reload=False)
    return None


def cmd_interfaces(args) -> dict:
    from ..capture import capture as capture_mod

    try:
        return {"interfaces": capture_mod.list_interfaces()}
    except capture_mod.CaptureError as exc:
        raise CliError(4, str(exc)) from exc


# --------------------------------------------------------------------------
# capture (foreground — the CLI has no daemon to hold a session)
# --------------------------------------------------------------------------
def cmd_capture(args) -> dict:
    from ..config import PCAPS_DIR
    from ..capture import capture as capture_mod

    if args.capture_cmd == "start":
        try:
            session = capture_mod.start_capture(
                args.iface,
                PCAPS_DIR,
                duration_seconds=args.seconds,
                packet_target=args.packets,
                buffer_mb=args.buffer_mb,
            )
        except capture_mod.CaptureError as exc:
            raise CliError(4, str(exc)) from exc
        try:
            while session.status() == "CAPTURING":
                time.sleep(0.5)
                session = capture_mod.check_and_finalize(session)
        except KeyboardInterrupt:
            session = capture_mod.stop_capture(session)
        if session.error:
            raise CliError(4, session.error)
        return session.to_dict()

    if args.capture_cmd == "packets":
        pcap = Path(args.pcap)
        if not pcap.is_file():
            raise CliError(3, f"PCAP not found: {pcap}")
        return {
            "pcap_path": str(pcap),
            "offset": args.offset,
            "limit": args.limit,
            "packets": capture_mod.read_packets(pcap, limit=args.limit, offset=args.offset),
        }

    if args.capture_cmd == "stat":
        pcap = Path(args.pcap)
        if not pcap.is_file():
            raise CliError(3, f"PCAP not found: {pcap}")
        try:
            return capture_mod.validate_and_stat_pcap(pcap)
        except capture_mod.CaptureError as exc:
            raise CliError(4, str(exc)) from exc

    raise CliError(2, f"unknown capture subcommand: {args.capture_cmd}")


# --------------------------------------------------------------------------
# extract / predict
# --------------------------------------------------------------------------
def cmd_extract(args) -> dict:
    from ..config import FEATURES_DIR
    from ..extraction.extract import run_extraction, ExtractionError

    pcap = Path(args.pcap)
    if not pcap.is_file():
        raise CliError(3, f"PCAP not found: {pcap}")
    features_dir = Path(args.features_dir) if args.features_dir else FEATURES_DIR
    try:
        return run_extraction(pcap, features_dir)
    except ExtractionError as exc:
        raise CliError(4, str(exc)) from exc


def cmd_predict(args) -> dict:
    from ..prediction.predict import predict_csv, PredictionError

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise CliError(3, f"CSV not found: {csv_path}")
    try:
        result = predict_csv(csv_path)
    except PredictionError as exc:
        raise CliError(4, str(exc)) from exc
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    # the per-flow list is huge; drop it from the returned view (the full
    # object is still written by --out).
    return {k: v for k, v in result.items() if k != "flows"}


def cmd_predict_metrics(args) -> dict:
    from ..prediction.metrics import evaluate_ann_metrics

    try:
        real = evaluate_ann_metrics(args.csv_path)
    except (FileNotFoundError, ValueError) as exc:
        raise CliError(4, str(exc)) from exc
    if real is None:
        raise CliError(
            3,
            "No ground-truth labels — set NIDS_CICIDS2017_DIR or pass "
            "--csv-path to a labelled CSV.",
        )
    return real


# --------------------------------------------------------------------------
# explain / SHAP
# --------------------------------------------------------------------------
_EXPLAIN_KINDS = {"ann": "ann", "lstm": "lstm", "hgb": "hist_gradient_boosting"}


def cmd_explain(args) -> dict:
    import numpy as np

    from ..prediction.explanation_runtime import submit_explanation
    from ..prediction.shap_service import jobs as explanation_jobs

    values_path = Path(args.values)
    if not values_path.is_file():
        raise CliError(3, f"values file not found: {values_path}")
    if values_path.suffix != ".npy":
        raise CliError(2, "--values must be a .npy array of already-scaled model input")
    values = np.load(values_path)
    model_kind = _EXPLAIN_KINDS[args.model]
    try:
        job = submit_explanation(model_kind, values.astype("float32"), args.explained_class)
    except (ValueError, RuntimeError, OSError) as exc:
        raise CliError(3, str(exc)) from exc
    if not args.wait:
        return job
    job_id = job["job_id"]
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        current = explanation_jobs.get(job_id) or job
        if current.get("status") in ("completed", "failed"):
            return current
        time.sleep(0.25)
    raise CliError(4, f"explanation job {job_id} did not finish within {args.timeout}s")


# --------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------
def cmd_temporal(args) -> dict:
    if args.temporal_cmd == "prepare":
        from ..temporal.temporal_dataset import prepare_temporal_dataset
        from ..temporal.windowing import TemporalError

        csv_path = Path(args.csv)
        if not csv_path.is_file():
            raise CliError(3, f"CSV not found: {csv_path}")
        out_dir = Path(args.out) if args.out else _repo_root() / "data" / "temporal" / csv_path.stem
        try:
            return prepare_temporal_dataset(csv_path, out_dir, args.window, args.seq)
        except TemporalError as exc:
            raise CliError(4, str(exc)) from exc

    if args.temporal_cmd == "validate":
        from ..temporal.validate import validate_temporal_dataset, ValidationError

        temporal_dir = Path(args.dir)
        source_csv = Path(args.source_csv)
        if not temporal_dir.is_dir():
            raise CliError(3, f"temporal dir not found: {temporal_dir}")
        if not source_csv.is_file():
            raise CliError(3, f"source CSV not found: {source_csv}")
        try:
            return validate_temporal_dataset(source_csv, temporal_dir)
        except ValidationError as exc:
            raise CliError(4, str(exc)) from exc

    raise CliError(2, f"unknown temporal subcommand: {args.temporal_cmd}")


# --------------------------------------------------------------------------
# one-step LSTM (Phase 3)
# --------------------------------------------------------------------------
def cmd_lstm(args) -> dict:
    if args.lstm_cmd == "train":
        from ..lstm.jobs import start_training

        try:
            return start_training(args.force)
        except RuntimeError as exc:
            raise CliError(4, str(exc)) from exc

    if args.lstm_cmd == "status":
        from ..lstm.jobs import read_status

        return read_status()

    if args.lstm_cmd == "forecast":
        from ..lstm.jobs import forecast_latest

        source = Path(args.windows) if args.windows else None
        try:
            return forecast_latest(windows_source=source)
        except RuntimeError as exc:
            raise CliError(3, str(exc)) from exc

    if args.lstm_cmd == "evaluate":
        from ..lstm.rigorous_evaluation import evaluate_saved_artifact

        try:
            return evaluate_saved_artifact(args.artifact_dir or None, None, None)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise CliError(3, str(exc)) from exc

    if args.lstm_cmd == "report":
        return _read_json(_repo_root() / "reports" / "lstm_evaluation_report.json")

    raise CliError(2, f"unknown lstm subcommand: {args.lstm_cmd}")


# --------------------------------------------------------------------------
# multi-step LSTM (Phase 4)
# --------------------------------------------------------------------------
def cmd_multistep(args) -> dict:
    reports = _repo_root() / "reports"
    if args.multistep_cmd == "dataset":
        from ..lstm_multistep.dataset import prepare_multistep_dataset

        _, manifest, _ = prepare_multistep_dataset(args.force)
        return {"splits": manifest.get("splits")}

    if args.multistep_cmd == "train":
        from ..lstm_multistep.training import train_multistep

        result = train_multistep(args.force)
        return {
            "model_version": result["evaluation"]["model_version"],
            "evaluation_status": result["evaluation"]["evaluation_status"],
        }

    if args.multistep_cmd == "evaluate":
        return _read_json(reports / "multistep_evaluation_report.json")

    if args.multistep_cmd == "benchmark":
        return _read_json(reports / "multistep_performance_benchmark.json")

    raise CliError(2, f"unknown multistep subcommand: {args.multistep_cmd}")


# --------------------------------------------------------------------------
# world model (K-step infiltration forecast)
# --------------------------------------------------------------------------
def cmd_worldmodel(args) -> dict:
    from ..worldmodel import jobs as worldmodel_jobs
    from ..worldmodel.engine import WorldModelUnavailable

    if args.worldmodel_cmd == "train":
        try:
            # A short-lived CLI process cannot reliably retain the spawned API
            # worker on every platform.  Train synchronously here; the API
            # continues to use its background job controller.
            from ..worldmodel.training import train_world_model
            return train_world_model(force_rebuild=args.force, allow_ungated=args.allow_ungated)
        except RuntimeError as exc:
            raise CliError(4, str(exc)) from exc

    if args.worldmodel_cmd == "status":
        return worldmodel_jobs.read_status()

    if args.worldmodel_cmd == "benchmark":
        from ..worldmodel.config import LATEST_PATH
        from ..lstm.config import repository_path
        if not LATEST_PATH.is_file():
            raise CliError(3, "No active world-model artifact/report exists.")
        latest = _read_json(LATEST_PATH)
        report = _read_json(repository_path(latest["artifact_dir"]) / "report.json")
        if "benchmark" not in report.get("test", {}):
            raise CliError(3, "Active world-model report has no benchmark block.")
        return report["test"]["benchmark"]

    if args.worldmodel_cmd == "forecast":
        source = Path(args.windows) if args.windows else None
        try:
            return worldmodel_jobs.forecast(windows_source=source, k=args.k)
        except WorldModelUnavailable as exc:
            raise CliError(3, str(exc)) from exc

    raise CliError(2, f"unknown worldmodel subcommand: {args.worldmodel_cmd}")


# --------------------------------------------------------------------------
# benchmark  (mirror of GET /api/benchmark)
# --------------------------------------------------------------------------
def _flat_metrics(metrics: dict) -> dict:
    attack = metrics.get("attack_forecasting", {}) or {}
    return {
        "macro_f1": metrics.get("macro_f1"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "weighted_f1": metrics.get("weighted_f1"),
        "attack_precision": attack.get("precision"),
        "attack_recall": attack.get("recall"),
        "attack_f1": attack.get("f1"),
        "attack_false_positive_rate": attack.get("false_positive_rate"),
    }


def cmd_benchmark(args) -> dict:
    report = _read_json(_repo_root() / "reports" / "lstm_evaluation_report.json")
    evals = report.get("evaluations", {})

    def pick(split: str) -> dict:
        block = evals.get(split, {})
        return {
            "lstm": _flat_metrics(block.get("lstm", {})),
            "logistic_regression": _flat_metrics(block.get("logistic_regression", {})),
        }

    return {
        "source": "reports/lstm_evaluation_report.json",
        "model_version": report.get("model_identity", {}).get("model_version")
        or report.get("model_version"),
        "headline_split": "validation",
        "one_step": {name: pick(name) for name in ("validation", "train", "test")},
        "note": "Frozen Phase 3 rolling-origin evaluation; validation split is the "
        "honest comparison (test split has no attack targets).",
    }


# --------------------------------------------------------------------------
# MITRE mapping
# --------------------------------------------------------------------------
def cmd_mitre(args) -> dict:
    from ..mitre.mapper import MitreAttackMapper, AttackMetadataError

    probabilities = _kv_pairs(args.prob, cast_float=True)
    if not probabilities:
        raise CliError(2, "at least one --prob CLASS=VALUE is required")
    features = _kv_pairs(args.feature, cast_float=True)
    try:
        mapper = MitreAttackMapper()
        return mapper.map_forecast(args.state, probabilities, features)
    except AttackMetadataError as exc:
        raise CliError(4, str(exc)) from exc


# --------------------------------------------------------------------------
# pipeline — on-disk artifact / report inventory
# --------------------------------------------------------------------------
def cmd_pipeline(args) -> dict:
    root = _repo_root()

    def exists(rel: str) -> bool:
        return (root / rel).exists()

    temporal_root = root / "data" / "temporal"
    datasets = sorted(p.name for p in temporal_root.glob("*")) if temporal_root.is_dir() else []
    lstm_root = root / "artifacts" / "lstm_forecaster"
    lstm_versions = sorted(p.name for p in lstm_root.glob("v1-*")) if lstm_root.is_dir() else []
    multistep_root = root / "artifacts" / "lstm_multistep"
    multistep_versions = sorted(p.name for p in multistep_root.glob("v1-*")) if multistep_root.is_dir() else []

    return {
        "temporal_datasets": datasets,
        "lstm_forecaster_versions": lstm_versions,
        "lstm_forecaster_active": exists("artifacts/lstm_forecaster/latest.json"),
        "lstm_multistep_versions": multistep_versions,
        "worldmodel_trained": exists("artifacts/worldmodel/latest.json"),
        "reports": {
            "lstm_evaluation": exists("reports/lstm_evaluation_report.json"),
            "multistep_evaluation": exists("reports/multistep_evaluation_report.json"),
            "multistep_benchmark": exists("reports/multistep_performance_benchmark.json"),
        },
        "shap_backgrounds": {
            "ann": exists("models/ann_shap_background.npy"),
        },
    }
