"""
Offline PCAP/CSV upload analysis: a second input mode alongside live
capture, sharing the same extraction/prediction logic (backend/extraction,
backend/prediction) rather than duplicating it. Live capture
(backend/capture/capture.py's start_capture/stop_capture, and
backend/api/main.py's PipelineState) is untouched by this module.

Each upload gets its own session directory and its own state machine:

  PCAP: FILE_UPLOADED -> VALIDATING -> EXTRACTING -> PREDICTING -> PREDICTION_COMPLETED
  CSV:  FILE_UPLOADED -> VALIDATING ->               PREDICTING -> PREDICTION_COMPLETED
  (either -> ERROR at any point)

The uploaded file is never overwritten: it's saved under session/input/
with a server-generated filename (the client's filename is never used to
build a filesystem path, only kept as a display string), and all derived
output (extracted/validated CSV with Current_State added, prediction
JSON) is written under session/features/ and session/results/.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import UPLOADS_DIR
from ..capture.capture import validate_and_stat_pcap, CaptureError
from ..extraction.extract import run_extraction, ExtractionError
from ..extraction.parallel_extract import run_parallel_extraction
from ..prediction.predict import predict_csv, PredictionError
from ..prediction.features import match_columns

logger = logging.getLogger("nids.upload")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

ALLOWED_EXTENSIONS = {".pcap", ".pcapng", ".csv"}


class UploadValidationError(Exception):
    """Raised for problems caught before/outside the shared pipeline
    modules (bad extension, empty file, path-unsafe name)."""
    pass


def _sanitize_display_name(name: str) -> str:
    """For display / download-filename use only — never used to build a
    filesystem path. Strips any directory components and unsafe chars."""
    base = Path(name or "upload").name
    base = re.sub(r"[^\w\-. ]", "_", base)
    return base or "upload"


@dataclass
class UploadSession:
    session_id: str
    original_filename: str
    input_type: str  # "pcap" | "csv"
    file_size: int
    stored_path: Path
    session_dir: Path
    features_dir: Path
    results_dir: Path
    created_at: float = field(default_factory=time.time)
    stage: str = "FILE_UPLOADED"
    error: Optional[str] = None
    packet_count: Optional[int] = None
    row_count: Optional[int] = None
    flow_count: Optional[int] = None
    extraction_result: Optional[dict] = None
    prediction_result: Optional[dict] = None
    processed_csv_path: Optional[str] = None
    processing_seconds: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_stage(self, stage: str):
        with self._lock:
            self.stage = stage
        logger.info(f"[Upload {self.session_id}] stage -> {stage}")

    def set_error(self, message: str):
        with self._lock:
            self.stage = "ERROR"
            self.error = message
        logger.info(f"[Upload {self.session_id}] ERROR: {message}")

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "session_id": self.session_id,
                "filename": self.original_filename,
                "input_type": self.input_type,
                "file_size": self.file_size,
                "stage": self.stage,
                "error": self.error,
                "packet_count": self.packet_count,
                "row_count": self.row_count,
                "flow_count": self.flow_count,
                "extraction": self.extraction_result,
                "prediction": self.prediction_result,
                "processed_csv_path": self.processed_csv_path,
                "processing_seconds": self.processing_seconds,
            }


_sessions: dict[str, UploadSession] = {}
_sessions_lock = threading.Lock()


def list_sessions() -> list[dict]:
    with _sessions_lock:
        sessions = list(_sessions.values())
    return sorted(
        (s.to_dict() for s in sessions),
        key=lambda d: d["session_id"], reverse=True,
    )


def get_session(session_id: str) -> Optional[UploadSession]:
    with _sessions_lock:
        return _sessions.get(session_id)


def create_upload_session(original_filename: str, file_bytes: bytes) -> UploadSession:
    """
    Validates extension + non-empty, allocates an isolated session
    directory, and saves the file under a server-generated name. Raises
    UploadValidationError for anything that should be rejected before a
    session/pipeline even starts.
    """
    ext = Path(original_filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported file type. Please upload a PCAP, PCAPNG, or "
            "CICFlowMeter-compatible CSV."
        )
    if not file_bytes:
        raise UploadValidationError("Uploaded file is empty.")

    session_id = f"upload_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_dir = UPLOADS_DIR / session_id
    input_dir = session_dir / "input"
    features_dir = session_dir / "features"
    results_dir = session_dir / "results"
    for d in (input_dir, features_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    stored_path = input_dir / f"{session_id}{ext}"  # server-controlled name, never the client's
    stored_path.write_bytes(file_bytes)

    input_type = "csv" if ext == ".csv" else "pcap"

    if input_type == "csv":
        # Cheap synchronous schema pre-check (header row only) so an
        # obviously-incompatible CSV gets a real 400 immediately instead
        # of only surfacing via polling — reuses the exact same matcher
        # predict_csv() uses later, so there's one source of truth for
        # what "missing features" means.
        try:
            header_df = pd.read_csv(stored_path, low_memory=False, nrows=0)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise UploadValidationError(f"Malformed CSV: {e}")
        try:
            match_columns(list(header_df.columns))
        except ValueError as e:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise UploadValidationError(str(e))

    session = UploadSession(
        session_id=session_id,
        original_filename=_sanitize_display_name(original_filename),
        input_type=input_type,
        file_size=len(file_bytes),
        stored_path=stored_path,
        session_dir=session_dir,
        features_dir=features_dir,
        results_dir=results_dir,
    )
    with _sessions_lock:
        _sessions[session_id] = session
    logger.info(
        f"[Upload {session_id}] Saved {input_type} upload "
        f"'{session.original_filename}' ({session.file_size} bytes)"
    )
    return session


def _finish(session: UploadSession, t0: float):
    session.processing_seconds = round(time.time() - t0, 3)


def process_pcap_upload(session: UploadSession):
    t0 = time.time()
    try:
        session.set_stage("VALIDATING")
        stats = validate_and_stat_pcap(session.stored_path)
        session.packet_count = stats.get("packet_count")
        logger.info(f"[Upload {session.session_id}] Valid PCAP, {session.packet_count} packets")

        session.set_stage("EXTRACTING")
        result = run_parallel_extraction(
            session.stored_path, session.features_dir, packet_count=session.packet_count
        )
        session.extraction_result = result
        session.flow_count = result["flow_count"]
        logger.info(
            f"[Upload {session.session_id}] Extracted {result['flow_count']} flows "
            f"({'parallel, %d chunks' % result['chunks'] if result.get('parallel') else 'serial'})"
        )

        session.set_stage("PREDICTING")
        pred = predict_csv(Path(result["output_csv"]))
        session.prediction_result = pred
        session.processed_csv_path = pred["output_csv"]

        _finish(session, t0)
        session.set_stage("PREDICTION_COMPLETED")
    except (CaptureError, ExtractionError, PredictionError) as e:
        _finish(session, t0)
        session.set_error(str(e))
    except Exception as e:
        _finish(session, t0)
        session.set_error(f"Unexpected error processing PCAP: {e}")


def process_csv_upload(session: UploadSession):
    t0 = time.time()
    try:
        session.set_stage("VALIDATING")
        try:
            df = pd.read_csv(session.stored_path, low_memory=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise UploadValidationError(f"Malformed CSV: {e}")

        if df.empty:
            raise UploadValidationError("Uploaded CSV has no data rows.")
        session.row_count = len(df)
        logger.info(f"[Upload {session.session_id}] Loaded CSV, {session.row_count} rows")

        # Never mutate the user's original upload: predict_csv writes
        # Current_State in place, so it runs against a working copy in
        # features/, leaving session/input/ untouched.
        working_csv = session.features_dir / f"{session.session_id}.csv"
        shutil.copy(session.stored_path, working_csv)

        session.set_stage("PREDICTING")
        # predict_csv() performs the actual schema validation (column
        # matching against the 77 trained features via features.py) as
        # its first real step; a ValueError there becomes a PredictionError
        # with the specific missing-feature names, which is exactly what
        # Test D expects.
        pred = predict_csv(working_csv)
        session.prediction_result = pred
        session.processed_csv_path = pred["output_csv"]
        session.flow_count = pred["flows_analyzed"]

        _finish(session, t0)
        session.set_stage("PREDICTION_COMPLETED")
    except (UploadValidationError, PredictionError) as e:
        _finish(session, t0)
        session.set_error(str(e))
    except Exception as e:
        _finish(session, t0)
        session.set_error(f"Unexpected error processing CSV: {e}")


def start_processing(session: UploadSession):
    target = process_pcap_upload if session.input_type == "pcap" else process_csv_upload
    thread = threading.Thread(target=target, args=(session,), daemon=True)
    thread.start()
