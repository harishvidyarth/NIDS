"""Reproducibility metadata and deterministic TensorFlow setup."""
from __future__ import annotations
import hashlib, os, platform
import numpy as np
import pandas as pd
import sklearn
import joblib
from .config import BATCH_SIZE, DEFAULT_K, HIDDEN_DIM, MAX_EPOCHS, SEED, SEQUENCE_LENGTH

def configure_determinism(seed=SEED):
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    return tf

def manifest_fingerprint(frames):
    h = hashlib.sha256()
    for frame in frames:
        h.update(str(getattr(frame, "shape", "")).encode())
        h.update(frame.to_csv(index=False).encode())
    return h.hexdigest()

def run_config(frames):
    import tensorflow as tf
    return {"seed": SEED, "sequence_length": SEQUENCE_LENGTH, "forecast_k": DEFAULT_K,
      "hidden_dimension": HIDDEN_DIM, "epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
      "loss_weights": {"class_probs": 1.0, "next_state": 0.3},
      "split_strategy": "per-session chronological segments", "boundary_policy": "deterministic chronological",
      "embargo_windows": 6, "dataset_cache_manifest_sha256": manifest_fingerprint(frames),
      "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__,
                   "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}}
