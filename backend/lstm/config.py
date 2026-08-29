from __future__ import annotations

import os
from pathlib import Path

from ..config import REPO_ROOT
from ..temporal.config import STATE_CLASSES

FORECAST_CLASSES = tuple(STATE_CLASSES)
WINDOW_SIZE_SECONDS = 10
SEQUENCE_LENGTH = 5
PROXY_CADENCE_SECONDS = 1
SCHEMA_VERSION = "row-order-proxy/v2"
SEED = 42
CHUNK_SIZE = 50_000
HOLDOUT_RATIO = 0.15
ROLLING_FOLDS = 3
MAX_EPOCHS = 40
BATCH_SIZE = 512

SOURCE_FILENAMES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
)

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "lstm_forecaster"
CACHE_ROOT = REPO_ROOT / "data" / "lstm_cache"
STATUS_PATH = ARTIFACT_ROOT / "job_status.json"
LATEST_PATH = ARTIFACT_ROOT / "latest.json"


def repository_path(path: Path | str) -> Path:
    """Resolve repository-relative pointers while accepting legacy absolute paths."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def repository_relative(path: Path | str) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.name


def source_dir() -> Path:
    configured = os.environ.get("NIDS_CICIDS2017_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = (
        REPO_ROOT.parent
        / "Project - Site"
        / "CICFlowMeter"
        / "cicflow-feature-extractor"
        / "training-data"
        / "cicids2017"
    )
    return candidate.resolve()


def source_paths(directory: Path | None = None) -> list[Path]:
    directory = Path(directory or source_dir())
    paths = [directory / name for name in SOURCE_FILENAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required CICIDS2017 source file(s) are missing. Set "
            "NIDS_CICIDS2017_DIR to their directory: " + ", ".join(missing)
        )
    return paths
