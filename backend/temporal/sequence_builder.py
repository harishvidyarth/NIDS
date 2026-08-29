"""
Builds the explicit state-transition table (S_t -> S_t+1) and the sliding
temporal sequences (S_t..S_t+L-1 -> S_t+L) used by a future forecasting
model. Sequences are built strictly in chronological window order — never
shuffled — and only from windows that are contiguous within a single
split (see splitting.py), so no sequence's target ever depends on a
future window that leaked in from a different split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STATE_CLASSES
from .schema import STATE_FEATURE_NAMES
from .windowing import TemporalError


def build_state_transitions(states_df: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive pair of windows (by window_id order, which
    is already the chronological order states were built in)."""
    if len(states_df) < 2:
        return pd.DataFrame(columns=[
            "current_window", "next_window", "current_state", "next_state",
            "current_attack_present", "next_attack_present", "time_delta_seconds",
        ])

    cur = states_df.iloc[:-1].reset_index(drop=True)
    nxt = states_df.iloc[1:].reset_index(drop=True)

    time_delta = (nxt["window_start"] - cur["window_start"]).dt.total_seconds()

    return pd.DataFrame({
        "current_window": cur["window_id"],
        "next_window": nxt["window_id"],
        "current_state": cur["dominant_state"],
        "next_state": nxt["dominant_state"],
        "current_attack_present": cur["attack_present"],
        "next_attack_present": nxt["attack_present"],
        "time_delta_seconds": time_delta,
    })


def raise_if_insufficient_windows(n_windows: int, sequence_length: int) -> None:
    required = sequence_length + 1
    if n_windows < required:
        raise TemporalError(
            "Insufficient temporal data.\n\n"
            f"Required:\n{required} windows\n\n"
            f"Available:\n{n_windows} windows\n\n"
            "Capture/upload a longer traffic sample."
        )


def build_sequences(states_df: pd.DataFrame, sequence_length: int) -> dict:
    """
    Sliding-window sequence construction over states_df, assumed to
    already be one contiguous chronological block (a single split).
    Raises TemporalError (not a silently-empty result) if there aren't
    enough windows — see config.MIN_WINDOWS_REQUIRED_FACTOR.

    Returns a dict of numpy arrays ready to np.savez():
      X                 (num_sequences, sequence_length, num_features) raw (unscaled) state vectors
      y_state_vector     (num_sequences, num_features) target window's raw state vector
      y_dominant_state    (num_sequences,) string target label
      y_attack_present     (num_sequences,) int target flag
      input_window_ids     (num_sequences, sequence_length) int
      target_window_id      (num_sequences,) int
    """
    n_windows = len(states_df)
    raise_if_insufficient_windows(n_windows, sequence_length)

    feature_matrix = states_df[STATE_FEATURE_NAMES].to_numpy(dtype=np.float64)
    window_ids = states_df["window_id"].to_numpy()
    dominant_states = states_df["dominant_state"].to_numpy()
    attack_present = states_df["attack_present"].to_numpy()

    n_sequences = n_windows - sequence_length
    X = np.empty((n_sequences, sequence_length, feature_matrix.shape[1]), dtype=np.float64)
    y_state_vector = np.empty((n_sequences, feature_matrix.shape[1]), dtype=np.float64)
    y_dominant_state = np.empty((n_sequences,), dtype=object)
    y_attack_present = np.empty((n_sequences,), dtype=np.int64)
    input_window_ids = np.empty((n_sequences, sequence_length), dtype=np.int64)
    target_window_id = np.empty((n_sequences,), dtype=np.int64)

    for i in range(n_sequences):
        X[i] = feature_matrix[i:i + sequence_length]
        target_idx = i + sequence_length
        y_state_vector[i] = feature_matrix[target_idx]
        y_dominant_state[i] = dominant_states[target_idx]
        y_attack_present[i] = attack_present[target_idx]
        input_window_ids[i] = window_ids[i:i + sequence_length]
        target_window_id[i] = window_ids[target_idx]

    return {
        "X": X,
        "y_state_vector": y_state_vector,
        "y_dominant_state": y_dominant_state,
        "y_attack_present": y_attack_present,
        "input_window_ids": input_window_ids,
        "target_window_id": target_window_id,
    }
