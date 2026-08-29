"""
Configuration for the temporal dataset / state-transition pipeline.
Nothing here trains a model — this module only prepares data for a
*future* forecasting phase (explicitly out of scope for this task).
"""
from __future__ import annotations

DEFAULT_WINDOW_SIZE_SECONDS = 10
DEFAULT_SEQUENCE_LENGTH = 5

# Chronological split ratios (NOT random — see splitting.py).
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Must match backend/prediction/predict.py's CLASS_NAMES exactly — the
# existing ANN's classes are the only valid state labels; this pipeline
# never invents new ones.
STATE_CLASSES = ["BENIGN", "DDoS", "DoS", "PortScan"]

MIN_WINDOWS_REQUIRED_FACTOR = 1  # windows must be >= sequence_length + 1
