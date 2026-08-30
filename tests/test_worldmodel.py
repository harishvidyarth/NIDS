"""World-model architecture + kill-chain mapping (backend/worldmodel/)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.worldmodel import config as C
from backend.worldmodel.killchain import (
    PROGRESS_STAGES,
    presentation_state,
    risk_level,
    state_to_stage,
)
from backend.worldmodel.model import build_world_model, rollout
from backend.worldmodel.training import _session_bucket, _session_frames_from_cache


def test_state_to_stage_covers_all_four_classes():
    assert state_to_stage("BENIGN") == "Benign"
    assert state_to_stage("PortScan") == "Reconnaissance"
    assert state_to_stage("DoS") == "Impact"
    assert state_to_stage("DDoS") == "Impact"


def test_presentation_state_ladder_and_terminal_alert():
    ps = presentation_state("Impact", risk="high")
    assert ps["predicted_stage"] == "Impact"
    assert ps["tactic_id"] == "TA0040"
    assert ps["terminal_impact_alert"] is True
    active = [s["name"] for s in ps["progress_stages"] if s["active"]]
    assert active == ["Impact"]
    assert [s["name"] for s in ps["progress_stages"]] == list(PROGRESS_STAGES)

    calm = presentation_state("Reconnaissance", risk="low")
    assert calm["terminal_impact_alert"] is False


def test_risk_level_bands():
    assert risk_level(0.1) == "low"
    assert risk_level(0.6) == "medium"
    assert risk_level(0.95) == "high"


def test_session_bucket_normalizes_csv_suffix():
    # session_id arrives as a bare stem from the window cache / builder,
    # while the *_SESSIONS tuples carry ".csv" — both must bucket the same.
    assert _session_bucket("Monday-WorkingHours.pcap_ISCX") == "train"
    assert _session_bucket("Monday-WorkingHours.pcap_ISCX.csv") == "train"
    assert _session_bucket("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX") == "validation"
    assert _session_bucket("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX") == "test"
    assert _session_bucket("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") == "test"


def test_session_frames_from_cache_are_complete_and_multiclass():
    frames = _session_frames_from_cache()
    if frames is None:
        pytest.skip("no complete data/lstm_cache/ window set on this machine")
    import pandas as pd

    from backend.temporal.schema import STATE_FEATURE_NAMES

    assert len(frames) == 8
    combined = pd.concat(frames, ignore_index=True)
    assert {"window_id", "dominant_state"}.issubset(combined.columns)
    assert all(f in combined.columns for f in STATE_FEATURE_NAMES)
    # the chronological split must yield more than one dominant class overall
    assert combined["dominant_state"].nunique() >= 2
    # buckets actually split
    buckets = {_session_bucket(str(f["session_id"].iloc[0])) for f in frames}
    assert {"train", "validation", "test"} <= buckets


def test_rollout_is_k_steps_and_autoregressive():
    model = build_world_model()
    seq = np.random.default_rng(0).random((1, C.SEQUENCE_LENGTH, C.INPUT_DIM)).astype("float32")
    probs, states, attn = rollout(model, seq, k_steps=6)
    assert probs.shape == (6, C.N_CLASSES)
    assert states.shape == (6, C.INPUT_DIM)
    assert attn.shape == (6, C.SEQUENCE_LENGTH)
    # each step's class distribution is a proper softmax
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-4)
