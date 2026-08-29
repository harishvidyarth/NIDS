from __future__ import annotations

from pathlib import Path

from ..config import REPO_ROOT
from ..lstm.config import BATCH_SIZE, CACHE_ROOT, FORECAST_CLASSES, SEED, SEQUENCE_LENGTH, source_dir

HORIZONS = 6
SCHEMA_VERSION = "direct-h1-h6-row-order-proxy/v1"
MAX_EPOCHS = 30
EARLY_WARNING_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)

SOURCE_FILENAMES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)
TRAIN_SESSIONS = SOURCE_FILENAMES[:5]
VALIDATION_SESSIONS = SOURCE_FILENAMES[5:7]
TEST_SESSIONS = SOURCE_FILENAMES[7:]

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "lstm_multistep"
LATEST_PATH = ARTIFACT_ROOT / "latest.json"
REPORT_ROOT = REPO_ROOT / "reports"
DATASET_MANIFEST = CACHE_ROOT / "multistep_dataset_manifest.json"


def source_paths(directory: Path | None = None) -> list[Path]:
    directory = Path(directory or source_dir())
    paths = [directory / name for name in SOURCE_FILENAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required CICIDS2017 source file(s) are missing: " + ", ".join(missing))
    return paths
