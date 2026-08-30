from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from ..config import REPO_ROOT
from ..lstm.config import FORECAST_CLASSES, LATEST_PATH, repository_path
from ..temporal.schema import STATE_FEATURE_NAMES
from .features import TRAINING_FEATURES
from .predict import MODEL_PATH, _load_artifacts
from .shap_service import gradient_explanation, gradient_input_fallback, jobs, tree_explanation


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gradient_with_fallback(model, background, values, names, class_index, class_name):
    try:
        return gradient_explanation(model, background, values, names, class_index, class_name)
    except Exception as error:
        return gradient_input_fallback(model, values, names, class_index, class_name, str(error))


def submit_explanation(model_kind: str, values: np.ndarray, explained_class: str | None = None) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if model_kind == "ann":
        model, _ = _load_artifacts()
        background_path = REPO_ROOT / "models" / "ann_shap_background.npy"
        if not background_path.is_file():
            raise RuntimeError("ANN SHAP background is unavailable; rebuild it from training-only labeled flows.")
        class_name = explained_class or FORECAST_CLASSES[int(np.argmax(model.predict(values, verbose=0)[0]))]
        class_index = list(FORECAST_CLASSES).index(class_name)
        background = np.load(background_path)
        work = lambda: _gradient_with_fallback(model, background, values, list(TRAINING_FEATURES), class_index, class_name)
        return jobs.submit(_file_hash(MODEL_PATH), values, {"kind": model_kind, "class": class_name}, work)

    if model_kind == "lstm":
        latest = json.loads(LATEST_PATH.read_text())
        artifact_dir = repository_path(latest["artifact_dir"])
        model_path = artifact_dir / "model.keras"
        background_path = artifact_dir / "shap_background.npy"
        if not background_path.is_file():
            raise RuntimeError("LSTM SHAP background is unavailable; retrain the active model.")
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        output = model.output["dominant_state"] if isinstance(model.output, dict) else model.output
        wrapper = tf.keras.Model(model.input, output)
        prediction = wrapper.predict(values, verbose=0)
        class_name = explained_class or FORECAST_CLASSES[int(np.argmax(prediction.reshape(len(values), -1)[0, :4]))]
        class_index = list(FORECAST_CLASSES).index(class_name)
        background = np.load(background_path)
        names = [f"t-{values.shape[1] - 1 - step}:{name}" for step in range(values.shape[1]) for name in STATE_FEATURE_NAMES]
        work = lambda: _gradient_with_fallback(wrapper, background, values, names, class_index, class_name)
        return jobs.submit(_file_hash(model_path), values, {"kind": model_kind, "class": class_name}, work)

    if model_kind == "hist_gradient_boosting":
        latest_path = REPO_ROOT / "artifacts" / "flow_challenger" / "latest.json"
        latest = json.loads(latest_path.read_text())
        artifact_dir = repository_path(latest["artifact_dir"])
        model_path = artifact_dir / "model.bin"
        model = joblib.load(model_path)
        background = np.load(artifact_dir / "shap_background.npy")
        class_name = explained_class or str(model.predict(values)[0])
        work = lambda: tree_explanation(model, background, values, list(TRAINING_FEATURES), class_name)
        return jobs.submit(_file_hash(model_path), values, {"kind": model_kind, "class": class_name}, work)

    raise ValueError("model_kind must be ann, lstm, or hist_gradient_boosting")
