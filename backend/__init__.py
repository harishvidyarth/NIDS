"""NIDS backend package.

Quiet the noisy native-level logging TensorFlow emits on import (oneDNN
notice, "Unable to register cuDNN/cuFFT/cuBLAS", TF-TRT warnings). These
are informational lines on a CPU-only host, not errors, but they clutter
the server log. Set before any submodule (and therefore any `import
tensorflow`) runs. `setdefault` so an operator can still override.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:  # absl is pulled in by TensorFlow; keep its logger quiet too
    from absl import logging as _absl_logging

    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:  # pragma: no cover - absl always present with TF, but be safe
    pass
