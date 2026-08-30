from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

SEED = 42
BACKGROUND_SIZE = 100


def stratified_background(
    values: np.ndarray, labels: np.ndarray, size: int = BACKGROUND_SIZE, seed: int = SEED
) -> np.ndarray:
    values = np.asarray(values)
    labels = np.asarray(labels)
    if len(values) != len(labels) or not len(values):
        raise ValueError("Training values and labels must be non-empty and aligned.")
    target = min(size, len(values))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    allocations = np.maximum(1, np.floor(target * counts / counts.sum()).astype(int))
    while allocations.sum() > target:
        index = int(np.argmax(allocations))
        if allocations[index] > 1:
            allocations[index] -= 1
        else:
            break
    while allocations.sum() < target:
        index = int(np.argmax(counts - allocations))
        allocations[index] += 1
    selected = []
    for label, allocation in zip(classes, allocations):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=min(allocation, len(candidates)), replace=False).tolist())
    if len(selected) < target:
        remaining = np.setdiff1d(np.arange(len(values)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=target - len(selected), replace=False).tolist())
    return values[np.sort(np.asarray(selected[:target], dtype=int))]


def _ranked(feature_names: list[str], values: np.ndarray) -> list[dict]:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    ranked = sorted(zip(feature_names, flattened), key=lambda item: (-abs(item[1]), item[0]))
    return [
        {"feature": feature, "contribution": float(contribution), "direction": "increase" if contribution >= 0 else "decrease"}
        for feature, contribution in ranked
    ]


def tree_explanation(model, background: np.ndarray, sample: np.ndarray, feature_names: list[str], explained_class: str) -> dict:
    import shap

    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    explanation = explainer(sample)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        class_index = list(model.classes_).index(explained_class)
        values = values[..., class_index]
        base_values = base_values[..., class_index]
    return {
        "method": "shap.TreeExplainer",
        "is_shap": True,
        "explained_class": explained_class,
        "base_value": float(np.asarray(base_values).reshape(-1)[0]),
        "feature_contributions": _ranked(feature_names, values[0]),
    }


def gradient_explanation(model, background: np.ndarray, sample: np.ndarray, feature_names: list[str], class_index: int, explained_class: str) -> dict:
    import shap

    explainer = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(sample)
    if isinstance(raw, list):
        values = np.asarray(raw[class_index])[0]
    else:
        values = np.asarray(raw)
        values = values[0, ..., class_index] if values.ndim >= sample.ndim + 1 else values[0]
    prediction = model.predict(background, verbose=0)
    if isinstance(prediction, dict):
        prediction = next(iter(prediction.values()))
    base_value = float(np.asarray(prediction).reshape(len(background), -1)[:, class_index].mean())
    return {
        "method": "shap.GradientExplainer",
        "is_shap": True,
        "explained_class": explained_class,
        "base_value": base_value,
        "feature_contributions": _ranked(feature_names, np.asarray(values).reshape(-1)),
    }


def gradient_input_fallback(model, sample: np.ndarray, feature_names: list[str], class_index: int, explained_class: str, error: str) -> dict:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        output = model(tensor, training=False)
        if isinstance(output, dict):
            output = next(iter(output.values()))
        score = tf.reshape(output, (tf.shape(output)[0], -1))[:, class_index]
    values = (tape.gradient(score, tensor) * tensor).numpy()[0]
    return {
        "method": "gradient_x_input_fallback",
        "is_shap": False,
        "explained_class": explained_class,
        "base_value": None,
        "fallback_reason": error,
        "feature_contributions": _ranked(feature_names, values),
    }


class ExplanationJobs:
    def __init__(self, workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nids-explain")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._cache: dict[str, str] = {}

    @staticmethod
    def cache_key(model_hash: str, values: np.ndarray, options: dict) -> str:
        digest = hashlib.sha256()
        digest.update(model_hash.encode())
        array = np.ascontiguousarray(values)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        digest.update(json.dumps(options, sort_keys=True).encode())
        return digest.hexdigest()

    def submit(self, model_hash: str, values: np.ndarray, options: dict, work: Callable[[], dict]) -> dict:
        key = self.cache_key(model_hash, values, options)
        with self._lock:
            cached_job = self._cache.get(key)
            if cached_job:
                return {**self._jobs[cached_job], "cache_hit": True}
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {"job_id": job_id, "status": "queued", "cache_key": key, "cache_hit": False}
            self._cache[key] = job_id

        def run() -> None:
            with self._lock:
                self._jobs[job_id]["status"] = "running"
            try:
                result = work()
                update = {"status": "completed", "result": result}
            except Exception as error:
                update = {"status": "failed", "error": str(error)}
            with self._lock:
                self._jobs[job_id].update(update)

        self._executor.submit(run)
        return dict(self._jobs[job_id])

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


jobs = ExplanationJobs()
