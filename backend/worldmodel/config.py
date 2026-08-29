from __future__ import annotations

from ..config import REPO_ROOT
from ..temporal.schema import STATE_FEATURE_NAMES
from ..temporal.config import STATE_CLASSES

INPUT_DIM = len(STATE_FEATURE_NAMES)          # 28
N_CLASSES = len(STATE_CLASSES)                # 4  (BENIGN, DDoS, DoS, PortScan)
FORECAST_CLASSES = tuple(STATE_CLASSES)
SEQUENCE_LENGTH = 5                            # windows fed in (matches one-step LSTM)
WINDOW_SECONDS = 10                           # each temporal window
DEFAULT_K = 6                                 # rollout steps if the caller gives none
MAX_K = 24
HIDDEN_DIM = 64
MAX_EPOCHS = 30
BATCH_SIZE = 256
SEED = 42
EARLY_WARNING_THRESHOLD = 0.60                # infiltration-probability alarm line

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "worldmodel"
LATEST_PATH = ARTIFACT_ROOT / "latest.json"
STATUS_PATH = ARTIFACT_ROOT / "job_status.json"
