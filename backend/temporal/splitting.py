"""
Chronological (never random) train/validation/test split, plus a
temporal-state scaler fit strictly on the training split. Splitting
happens on WINDOWS first, then sequences are built independently within
each contiguous window block — never by slicing a globally-built sequence
array — so no sequence's input or target window ever crosses a split
boundary.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from .schema import STATE_FEATURE_NAMES
from .sequence_builder import build_sequences
from .windowing import TemporalError


def chronological_split(states_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    n = len(states_df)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    # remainder goes to test, so rounding never drops/duplicates a window
    n_test = n - n_train - n_val

    train_df = states_df.iloc[:n_train].reset_index(drop=True)
    val_df = states_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = states_df.iloc[n_train + n_val:].reset_index(drop=True)

    def bounds(d):
        if d.empty:
            return None
        return {
            "start": str(d["window_start"].iloc[0]),
            "end": str(d["window_end"].iloc[-1]),
            "n_windows": len(d),
            "window_id_range": [int(d["window_id"].iloc[0]), int(d["window_id"].iloc[-1])],
        }

    metadata = {
        "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO,
        "total_windows": n,
        "train": bounds(train_df), "validation": bounds(val_df), "test": bounds(test_df),
    }
    return train_df, val_df, test_df, metadata


def fit_scaler_on_train(train_df: pd.DataFrame):
    """MinMaxScaler, matching the convention already used for the ANN's
    own feature scaling (models/minmax.bin) — fit ONLY on the training
    windows, per the task's explicit no-leakage requirement."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[STATE_FEATURE_NAMES].to_numpy(dtype=np.float64))
    return scaler


def build_split_sequences(df: pd.DataFrame, sequence_length: int, scaler, split_name: str) -> dict:
    """Builds sequences for one split from its raw (unscaled) states, then
    scales X and y_state_vector with the already-fitted scaler. Raises
    TemporalError with the split name if that split doesn't have enough
    windows — this is expected and acceptable for a short capture: not
    every split is guaranteed enough data, and the caller may treat a
    too-small val/test split as informational rather than fatal."""
    try:
        seq = build_sequences(df, sequence_length)
    except TemporalError as e:
        raise TemporalError(f"[{split_name} split] {e}")

    n_seq, seq_len, n_feat = seq["X"].shape
    X_flat = seq["X"].reshape(-1, n_feat)
    X_scaled = scaler.transform(X_flat).reshape(n_seq, seq_len, n_feat)
    y_scaled = scaler.transform(seq["y_state_vector"])

    seq["X_scaled"] = X_scaled
    seq["y_state_vector_scaled"] = y_scaled
    return seq
