"""
FastAPI backend wiring the real pipeline: PCAP capture -> CICFlowMeter
extraction -> ANN prediction. No mock data anywhere — every field the UI
displays either comes from a real subprocess/model result or is null/an
explicit error.

State machine:
  IDLE -> CAPTURING -> CAPTURE_COMPLETED -> EXTRACTING -> EXTRACTION_COMPLETED
       -> PREDICTING -> PREDICTION_COMPLETED
  any state -> ERROR on failure
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from ..config import PCAPS_DIR, FEATURES_DIR, RESULTS_DIR, REPO_ROOT
from ..capture import capture as capture_mod
from ..extraction.extract import run_extraction, ExtractionError
from ..prediction.predict import predict_csv, PredictionError
from ..upload import manager as upload_mgr
from ..temporal.temporal_dataset import prepare_temporal_dataset
from ..temporal.windowing import TemporalError
from ..temporal.config import DEFAULT_SEQUENCE_LENGTH, DEFAULT_WINDOW_SIZE_SECONDS
from ..temporal.validate import validate_temporal_dataset, ValidationError
from ..lstm.config import LATEST_PATH
from ..lstm.jobs import forecast_latest, read_status as read_lstm_status, start_training
from ..lstm_multistep.training import forecast_latest as forecast_multistep_latest

app = FastAPI(title="NIDS Pipeline API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stage = "IDLE"
        self.error: Optional[str] = None
        self.capture_session: Optional[capture_mod.CaptureSession] = None
        self.extraction_result: Optional[dict] = None
        self.prediction_result: Optional[dict] = None
        self.timings: dict = {}

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "stage": self.stage,
                "error": self.error,
                "capture": self.capture_session.to_dict() if self.capture_session else None,
                "extraction": self.extraction_result,
                "prediction": self.prediction_result,
                "timings": self.timings,
            }

    def set_error(self, message: str):
        with self.lock:
            self.stage = "ERROR"
            self.error = message


state = PipelineState()


class StartCaptureRequest(BaseModel):
    interface: str
    # Upper-bound elapsed seconds (dumpcap's own `-a duration:N`) so a
    # capture on a quiet interface can't run forever waiting to reach
    # packet_target. Defaults to 300s — wide enough to realistically
    # reach the default 10000-packet target on a normal interface.
    duration_seconds: Optional[int] = capture_mod.DEFAULT_CAPTURE_DURATION_SECONDS
    # Primary capture target, distinct from the packet-table PAGE size —
    # this bounds what dumpcap/tcpdump actually captures (via their own
    # -c flag). Defaults to 10000, so the capture keeps going up to that
    # many packets rather than stopping early.
    packet_target: Optional[int] = capture_mod.DEFAULT_CAPTURE_PACKET_TARGET


class ExtractRequest(BaseModel):
    pcap_path: Optional[str] = None


class PredictRequest(BaseModel):
    csv_path: Optional[str] = None


@app.get("/api/interfaces")
def get_interfaces():
    try:
        return {"interfaces": capture_mod.list_interfaces()}
    except capture_mod.CaptureError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/capture/start")
def start_capture(req: StartCaptureRequest):
    with state.lock:
        if state.stage == "CAPTURING":
            raise HTTPException(status_code=409, detail="Capture already running.")
        state.error = None
    try:
        session = capture_mod.start_capture(
            req.interface,
            PCAPS_DIR,
            duration_seconds=req.duration_seconds,
            packet_target=req.packet_target,
        )
    except capture_mod.CaptureError as e:
        state.set_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    with state.lock:
        state.capture_session = session
        state.stage = "CAPTURING"
        state.extraction_result = None
        state.prediction_result = None
    return state.snapshot()


@app.post("/api/capture/stop")
def stop_capture():
    with state.lock:
        session = state.capture_session
    if session is None or session.status() != "CAPTURING":
        raise HTTPException(status_code=409, detail="No active capture to stop.")
    session = capture_mod.stop_capture(session)
    with state.lock:
        state.capture_session = session
        if session.error:
            state.stage = "ERROR"
            state.error = session.error
        else:
            state.stage = "CAPTURE_COMPLETED"
    return state.snapshot()


def _sync_capture_state():
    """Called on every status/pipeline poll while a capture is running:
    detects (a) dumpcap/tcpdump having exited on its own after reaching
    duration_target and/or packet_target, (b) duration_target elapsing on
    a platform with no native duration stop, and (c) the safety timeout —
    none of which anything else is watching for in the background — and
    finalizes the session exactly as a manual Stop would. A no-op
    otherwise."""
    with state.lock:
        session = state.capture_session
        stage = state.stage
    if session is None or stage != "CAPTURING":
        return
    updated = capture_mod.check_and_finalize(session)
    if updated.status() == "CAPTURING":
        return
    with state.lock:
        state.capture_session = updated
        if updated.error:
            state.stage = "ERROR"
            state.error = updated.error
        else:
            state.stage = "CAPTURE_COMPLETED"


@app.get("/api/capture/status")
def capture_status():
    _sync_capture_state()
    with state.lock:
        return state.capture_session.to_dict() if state.capture_session else {"status": "IDLE"}


@app.get("/api/capture/packets")
def capture_packets(offset: int = 0, limit: int = 100):
    """Paginated — offset/limit are pushed down to tshark itself (see
    capture.read_packets), so this never loads an entire large capture
    into memory just to slice a page out of it. `total_packets` is the
    real count from the capture session (capinfos), so the frontend can
    build "Rows X-Y of Z" pagination without guessing."""
    with state.lock:
        session = state.capture_session
    if session is None:
        return {"packets": [], "total_packets": None, "offset": offset, "limit": limit}
    packets = capture_mod.read_packets(session.pcap_path, limit=limit, offset=offset)
    return {
        "packets": packets,
        "total_packets": session.packet_count,
        "offset": offset,
        "limit": limit,
    }


def _run_extraction_bg(pcap_path: Path):
    try:
        with state.lock:
            state.stage = "EXTRACTING"
            state.error = None
        result = run_extraction(pcap_path, FEATURES_DIR)
        with state.lock:
            state.extraction_result = result
            state.stage = "EXTRACTION_COMPLETED"
    except ExtractionError as e:
        state.set_error(str(e))


@app.post("/api/extract")
def extract(req: ExtractRequest):
    with state.lock:
        if state.stage == "EXTRACTING":
            raise HTTPException(status_code=409, detail="Extraction already running.")
        pcap_path = req.pcap_path
        if not pcap_path:
            if not state.capture_session:
                raise HTTPException(status_code=400, detail="No captured PCAP available.")
            pcap_path = str(state.capture_session.pcap_path)

    thread = threading.Thread(target=_run_extraction_bg, args=(Path(pcap_path),), daemon=True)
    thread.start()
    return {"started": True, "pcap_path": pcap_path}


@app.get("/api/extract/status")
def extract_status():
    with state.lock:
        return {
            "stage": state.stage,
            "result": state.extraction_result,
            "error": state.error if state.stage == "ERROR" else None,
        }


def _run_prediction_bg(csv_path: Path):
    try:
        with state.lock:
            state.stage = "PREDICTING"
            state.error = None
        result = predict_csv(csv_path)
        with state.lock:
            state.prediction_result = result
            state.stage = "PREDICTION_COMPLETED"
        result_path = RESULTS_DIR / f"{csv_path.stem}_prediction.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        import json
        result_path.write_text(json.dumps(result, indent=2))
    except PredictionError as e:
        state.set_error(str(e))


@app.post("/api/predict")
def predict(req: PredictRequest):
    with state.lock:
        if state.stage == "PREDICTING":
            raise HTTPException(status_code=409, detail="Prediction already running.")
        csv_path = req.csv_path
        if not csv_path:
            if not state.extraction_result:
                raise HTTPException(status_code=400, detail="No extracted CSV available.")
            csv_path = state.extraction_result["output_csv"]

    thread = threading.Thread(target=_run_prediction_bg, args=(Path(csv_path),), daemon=True)
    thread.start()
    return {"started": True, "csv_path": csv_path}


@app.get("/api/predict/status")
def predict_status():
    with state.lock:
        return {
            "stage": state.stage,
            "result": state.prediction_result,
            "error": state.error if state.stage == "ERROR" else None,
        }


@app.get("/api/pipeline")
def pipeline_state():
    _sync_capture_state()
    return state.snapshot()


@app.post("/api/pipeline/reset")
def pipeline_reset():
    with state.lock:
        if state.stage in ("CAPTURING", "EXTRACTING", "PREDICTING"):
            raise HTTPException(status_code=409, detail=f"Cannot reset while {state.stage}.")
        state.stage = "IDLE"
        state.error = None
        state.capture_session = None
        state.extraction_result = None
        state.prediction_result = None
    # Resetting the pipeline also clears downstream temporal/validation
    # state (both were derived from the capture this reset is clearing) —
    # this does not delete any file on disk, only the in-memory status the
    # UI polls.
    with temporal_state.lock:
        if temporal_state.stage != "PREPARING":
            temporal_state.stage = "IDLE"
            temporal_state.error = None
            temporal_state.summary = None
    with validation_state.lock:
        if validation_state.stage != "VALIDATING":
            validation_state.stage = "NOT_VALIDATED"
            validation_state.error = None
            validation_state.report = None
    return state.snapshot()


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Offline analysis input mode, independent of live capture (does not
    touch PipelineState/`state` above and never starts dumpcap). Accepts
    .pcap/.pcapng/.csv, validates, stores the upload under an isolated
    session directory, and starts the appropriate background pipeline
    (PCAP: validate -> extract -> predict; CSV: validate -> predict).
    """
    content = await file.read()
    try:
        session = upload_mgr.create_upload_session(file.filename, content)
    except upload_mgr.UploadValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    upload_mgr.start_processing(session)

    return {
        "success": True,
        "session_id": session.session_id,
        "filename": session.original_filename,
        "input_type": session.input_type,
        "status": "PROCESSING",
    }


@app.get("/api/upload/sessions")
def upload_sessions():
    return {"sessions": upload_mgr.list_sessions()}


@app.get("/api/upload/{session_id}/status")
def upload_status(session_id: str):
    session = upload_mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session.to_dict()


@app.get("/api/upload/{session_id}/download")
def upload_download(session_id: str):
    session = upload_mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    if session.stage != "PREDICTION_COMPLETED" or not session.processed_csv_path:
        raise HTTPException(
            status_code=409,
            detail=f"Processed CSV not ready yet (stage: {session.stage}).",
        )
    processed = Path(session.processed_csv_path)
    if not processed.exists():
        raise HTTPException(status_code=404, detail="Processed CSV file is missing on disk.")

    stem = Path(session.original_filename).stem
    suffix = "_features_prediction.csv" if session.input_type == "pcap" else "_prediction.csv"
    download_name = upload_mgr._sanitize_display_name(f"{stem}{suffix}")
    return FileResponse(
        path=str(processed), filename=download_name, media_type="text/csv"
    )


class TemporalRequest(BaseModel):
    csv_path: str
    window_size_seconds: int = DEFAULT_WINDOW_SIZE_SECONDS
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH


class TemporalState:
    """Separate, independent state — does not touch PipelineState/`state`
    or upload_mgr's sessions. Never trains anything; only prepares the
    windowed/sequenced dataset a future forecasting phase will consume."""
    def __init__(self):
        self.lock = threading.Lock()
        self.stage = "IDLE"
        self.error: Optional[str] = None
        self.summary: Optional[dict] = None

    def snapshot(self) -> dict:
        with self.lock:
            if self.summary is None:
                json_summary = None
            else:
                # states_df/transitions_df/sequences carry DataFrames/ndarrays
                # not meant for JSON — the real artifacts are the files on
                # disk; this endpoint reports the same scalars CLI prints.
                json_summary = {k: v for k, v in self.summary.items()
                                 if k not in ("states_df", "transitions_df", "sequences")}
            return {"stage": self.stage, "error": self.error, "result": json_summary}


temporal_state = TemporalState()


def _run_temporal_bg(csv_path: Path, window_size_seconds: int, sequence_length: int):
    try:
        with temporal_state.lock:
            temporal_state.stage = "PREPARING"
            temporal_state.error = None
        output_dir = REPO_ROOT / "data" / "temporal" / csv_path.stem
        summary = prepare_temporal_dataset(csv_path, output_dir, window_size_seconds, sequence_length)
        with temporal_state.lock:
            temporal_state.summary = summary
            temporal_state.stage = "COMPLETED"
    except TemporalError as e:
        with temporal_state.lock:
            temporal_state.stage = "ERROR"
            temporal_state.error = str(e)


@app.post("/api/temporal/prepare")
def temporal_prepare(req: TemporalRequest):
    with temporal_state.lock:
        if temporal_state.stage == "PREPARING":
            raise HTTPException(status_code=409, detail="Temporal preparation already running.")
    csv_path = Path(req.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {csv_path}")
    thread = threading.Thread(
        target=_run_temporal_bg, args=(csv_path, req.window_size_seconds, req.sequence_length), daemon=True
    )
    thread.start()
    return {"started": True, "csv_path": str(csv_path)}


@app.get("/api/temporal/status")
def temporal_status():
    return temporal_state.snapshot()


class ValidationState:
    """Independent of TemporalState — validates an already-prepared
    temporal dataset (read-only) rather than building one."""
    def __init__(self):
        self.lock = threading.Lock()
        self.stage = "NOT_VALIDATED"  # NOT_VALIDATED | VALIDATING | VALIDATED | VALIDATED_WITH_WARNINGS | VALIDATION_FAILED | ERROR
        self.error: Optional[str] = None
        self.report: Optional[dict] = None

    def snapshot(self) -> dict:
        with self.lock:
            return {"stage": self.stage, "error": self.error, "report": self.report}


validation_state = ValidationState()


def _stage_for_overall_status(overall: str) -> str:
    return {
        "PASS": "VALIDATED",
        "WARNING": "VALIDATED_WITH_WARNINGS",
        "FAIL": "VALIDATION_FAILED",
        "NOT_AVAILABLE": "VALIDATION_FAILED",
    }.get(overall, "VALIDATION_FAILED")


def _run_validation_bg(source_csv: Path, temporal_dir: Path):
    try:
        with validation_state.lock:
            validation_state.stage = "VALIDATING"
            validation_state.error = None
        report = validate_temporal_dataset(source_csv, temporal_dir)
        report_path = temporal_dir / "validation_report.json"
        import json
        report_path.write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(report_path)
        with validation_state.lock:
            validation_state.report = report
            validation_state.stage = _stage_for_overall_status(report["overall_status"])
    except ValidationError as e:
        with validation_state.lock:
            validation_state.stage = "ERROR"
            validation_state.error = str(e)


@app.post("/api/temporal/validate")
def temporal_validate():
    with validation_state.lock:
        if validation_state.stage == "VALIDATING":
            raise HTTPException(status_code=409, detail="Validation already running.")
    with temporal_state.lock:
        if temporal_state.stage != "COMPLETED" or not temporal_state.summary:
            raise HTTPException(
                status_code=400,
                detail="TEMPORAL DATASET NOT AVAILABLE — run /api/temporal/prepare first.",
            )
        source_csv = Path(temporal_state.summary["input_csv"])
        temporal_dir = Path(temporal_state.summary["output_dir"])

    thread = threading.Thread(target=_run_validation_bg, args=(source_csv, temporal_dir), daemon=True)
    thread.start()
    return {"started": True, "source_csv": str(source_csv), "temporal_dir": str(temporal_dir)}


@app.get("/api/temporal/validate/status")
def temporal_validate_status():
    return validation_state.snapshot()


@app.get("/api/temporal/validate/report")
def temporal_validate_report():
    with validation_state.lock:
        report = validation_state.report
    if not report or "report_path" not in report:
        raise HTTPException(status_code=404, detail="No validation report available yet.")
    report_path = Path(report["report_path"])
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Validation report file is missing on disk.")
    return FileResponse(path=str(report_path), filename=report_path.name, media_type="application/json")


class LstmTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force_rebuild: bool = False


@app.post("/api/lstm/train")
def lstm_train(req: LstmTrainRequest):
    try:
        return start_training(req.force_rebuild)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/api/lstm/status")
def lstm_status():
    return read_lstm_status()


@app.post("/api/lstm/forecast")
def lstm_forecast():
    try:
        return forecast_latest()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/api/lstm/forecast/multistep")
def lstm_forecast_multistep():
    try:
        return forecast_multistep_latest()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/api/lstm/report")
def lstm_report():
    if not LATEST_PATH.is_file():
        raise HTTPException(status_code=404, detail="No completed LSTM report is available.")
    import json
    latest = json.loads(LATEST_PATH.read_text())
    report_path = Path(latest["report_path"])
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="The completed LSTM report is missing on disk.")
    return FileResponse(path=str(report_path), filename=report_path.name, media_type="application/json")


frontend_dir = REPO_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
