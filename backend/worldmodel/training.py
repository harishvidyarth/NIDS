"""Train the K-step infiltration world model on the CICIDS2017 temporal
windows (reuses backend.lstm_multistep's dataset preparation for the
per-session window frames, then builds (X, next-class, next-state)
sequences for the dual head).

Data-gated: needs the CICIDS2017 CSVs (NIDS_CICIDS2017_DIR). Without them
`prepare_multistep_dataset` raises FileNotFoundError and this returns a
clear message rather than a stack trace.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from ..lstm.config import repository_relative
from ..lstm_multistep.config import SOURCE_FILENAMES, TEST_SESSIONS, TRAIN_SESSIONS, VALIDATION_SESSIONS
from ..lstm_multistep.dataset import prepare_multistep_dataset
from ..temporal.schema import STATE_FEATURE_NAMES
from .config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    FORECAST_CLASSES,
    LATEST_PATH,
    MAX_EPOCHS,
    SEED,
    SEQUENCE_LENGTH,
)
from .model import build_world_model

_CLASS_IDX = {c: i for i, c in enumerate(FORECAST_CLASSES)}


def _norm_session(name: str) -> str:
    return name[:-4] if name.endswith(".csv") else name


_TRAIN_STEMS = {_norm_session(s) for s in TRAIN_SESSIONS}
_VAL_STEMS = {_norm_session(s) for s in VALIDATION_SESSIONS}
_TEST_STEMS = {_norm_session(s) for s in TEST_SESSIONS}


def _session_bucket(session_id: str) -> str:
    # session_id comes through as a bare stem (no .csv) from
    # prepare_session_windows / the cache loader, while the *_SESSIONS
    # tuples carry the .csv suffix — normalise both sides so the
    # chronological train/val/test split actually applies.
    stem = _norm_session(str(session_id))
    if stem in _VAL_STEMS:
        return "validation"
    if stem in _TEST_STEMS:
        return "test"
    return "train"  # train stems + anything unrecognised


def _session_frames_from_cache():
    """Assemble the 8 per-session windowed frames straight from the
    already-built window caches under data/lstm_cache/, using the exact
    cache_key set recorded in multistep_dataset_manifest.json (the set the
    working multi-step artifact was built from — its ground-truth labels
    carry PortScan/DoS/DDoS, unlike a fresh rebuild under the current
    Infiltration-drops-to-BENIGN label map). No raw CICIDS2017 CSV read.
    Returns None if the manifest or any of the 8 caches is missing."""
    import pandas as pd

    from ..lstm.config import CACHE_ROOT
    from ..temporal.schema import STATE_FEATURE_NAMES

    manifest_path = CACHE_ROOT / "multistep_dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        cache_entries = json.loads(manifest_path.read_text()).get("cache", [])
    except (OSError, ValueError):
        return None

    by_name = {}
    for entry in cache_entries:
        name = (entry.get("identity", {}).get("source", {}) or {}).get("name") or entry.get("source_path")
        if name:
            by_name[name] = entry.get("cache_key")

    frames = []
    for name in SOURCE_FILENAMES:
        key = by_name.get(name)
        if not key:
            return None
        npz_path = CACHE_ROOT / key / "windows.npz"
        if not npz_path.is_file():
            return None
        data = np.load(npz_path, allow_pickle=True)  # our own local cache; label columns are object arrays
        frame = pd.DataFrame({col: data[col] for col in data.files})
        if not {"window_id", "dominant_state"}.issubset(frame.columns) or any(
            f not in frame.columns for f in STATE_FEATURE_NAMES
        ):
            return None
        frame.insert(0, "session_id", _norm_session(name))
        frames.append(frame)
    return frames


def _build_pairs(frames, scaler=None):
    """(X_scaled [N, L, 28], y_class [N], y_state [N, 28], scaler) from
    per-session window frames, sliding by 1 within each contiguous
    window_id run."""
    L = SEQUENCE_LENGTH
    raw_X, y_class, raw_state = [], [], []
    for frame in frames:
        block = frame.sort_values("window_id").reset_index(drop=True)
        wid = block["window_id"].to_numpy(dtype=np.int64)
        feats = block[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        labels = block["dominant_state"].to_numpy(dtype=str)
        for s in range(len(block) - L):
            if not np.all(np.diff(wid[s : s + L + 1]) == 1):
                continue
            tgt = labels[s + L]
            if tgt not in _CLASS_IDX:
                continue
            raw_X.append(feats[s : s + L])
            y_class.append(_CLASS_IDX[tgt])
            raw_state.append(feats[s + L])
    if not raw_X:
        return None
    raw_X = np.stack(raw_X)
    raw_state = np.stack(raw_state)
    flat = raw_X.reshape(-1, raw_X.shape[-1])
    if scaler is None:
        scaler = MinMaxScaler().fit(flat)
    X_scaled = scaler.transform(flat).reshape(raw_X.shape).astype(np.float32)
    y_state = scaler.transform(raw_state).astype(np.float32)
    return X_scaled, np.asarray(y_class, dtype=np.int64), y_state, scaler


def train_world_model(force_rebuild: bool = False, status=lambda **k: None) -> dict:
    status(stage="preparing-dataset")

    # The K-step world-model only consumes the raw per-session window frames
    # (`prepare_multistep_dataset`'s 3rd return value). Those already exist
    # on disk under data/lstm_cache/ (the exact set the working multi-step
    # artifact was built from), so prefer them: it avoids the CICIDS2017 CSV
    # dependency AND the current label map's validate_class_support() /
    # object-array cache-reader breakage in prepare_multistep_dataset.
    session_frames = None if force_rebuild else _session_frames_from_cache()
    if session_frames is not None:
        status(stage="preparing-dataset", cache_state="hit", source="lstm_cache_manifest")
    else:
        try:
            # forward the dataset layer's own progress kwargs unchanged — it
            # already emits stage= ("ground_truth_windowing" / "cache"), so
            # re-injecting stage= here would collide.
            _dataset, _manifest, session_frames = prepare_multistep_dataset(
                force_rebuild=force_rebuild,
                progress=lambda **k: status(**k),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            session_frames = _session_frames_from_cache()
            if session_frames is None:
                raise RuntimeError(
                    "World-model training needs the CICIDS2017 CSVs or a complete "
                    "data/lstm_cache/ window set (per multistep_dataset_manifest.json). "
                    + str(exc)
                ) from exc
            status(stage="preparing-dataset", cache_state="hit", source="lstm_cache_manifest")

    buckets = {"train": [], "validation": [], "test": []}
    for frame in session_frames:
        sid = (
            str(frame["session_id"].iloc[0])
            if "session_id" in frame.columns and len(frame)
            else ""
        )
        buckets[_session_bucket(sid)].append(frame)

    train_pairs = _build_pairs(buckets["train"])
    if train_pairs is None:
        raise RuntimeError("No usable training window sequences were built.")
    Xtr, ytr_cls, ytr_state, scaler = train_pairs

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = build_world_model()
    model.compile(
        optimizer="adam",
        loss={"class_probs": "sparse_categorical_crossentropy", "next_state": "mse"},
        loss_weights={"class_probs": 1.0, "next_state": 0.3},
        metrics={"class_probs": "accuracy"},
    )

    val_pairs = _build_pairs(buckets["validation"], scaler)
    val_data = None
    if val_pairs is not None:
        Xva, yva_cls, yva_state, _ = val_pairs
        val_data = (Xva, {"class_probs": yva_cls, "next_state": yva_state})

    status(stage="fitting")
    model.fit(
        Xtr, {"class_probs": ytr_cls, "next_state": ytr_state},
        validation_data=val_data,
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
    )

    evaluation_status = "UNVERIFIED"
    test_pairs = _build_pairs(buckets["test"], scaler)
    if test_pairs is not None:
        Xte, yte_cls, _yte_state, _ = test_pairs
        probs, _ = model.predict(Xte, verbose=0)
        pred = np.argmax(probs, axis=1)
        macro = float(f1_score(
            yte_cls, pred, average="macro",
            labels=list(range(len(FORECAST_CLASSES))), zero_division=0,
        ))
        test_report = {
            "n": int(len(yte_cls)),
            "macro_f1": macro,
            "class_support": {c: int(np.sum(yte_cls == i)) for i, c in enumerate(FORECAST_CLASSES)},
        }
        evaluation_status = "VALIDATED" if macro >= 0.40 else "VALIDATED_WITH_WARNINGS"
    else:
        test_report = {"note": "no held-out attack sessions available for evaluation"}

    ts = time.strftime("%Y-%m-%d_%H%M%S")
    artifact_dir = ARTIFACT_ROOT / ts
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / "model.keras")
    joblib.dump(scaler, artifact_dir / "scaler.bin")
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_order": list(STATE_FEATURE_NAMES),
        "classes": list(FORECAST_CLASSES),
        "train_examples": int(len(Xtr)),
        "evaluation_status": evaluation_status,
        "test": test_report,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))
    LATEST_PATH.write_text(json.dumps({
        "artifact_dir": repository_relative(artifact_dir),
        "model_version": ts,
        "created_at": report["created_at"],
        "evaluation_status": evaluation_status,
    }, indent=2))
    status(stage="completed", evaluation_status=evaluation_status)
    return report
