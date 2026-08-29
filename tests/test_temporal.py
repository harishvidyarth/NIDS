"""
Automated tests for backend/temporal — covers timestamp parsing,
chronological sorting, windowing, aggregation, labeling, transitions,
sequences, chronological splitting, scaler leakage prevention, and the
insufficient-data / invalid-timestamp error paths (task section 24,
items 1-12).

Uses small synthetic-but-realistic frames built directly in the CICFlowMeter
snake_case schema this project's own extraction actually produces
(confirmed against a real generated CSV) — not fabricated column names.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.temporal.windowing import assign_windows, parse_timestamps, sort_chronologically, TemporalError
from backend.temporal.state_builder import build_temporal_states
from backend.temporal.sequence_builder import build_sequences, build_state_transitions, raise_if_insufficient_windows
from backend.temporal.splitting import chronological_split, fit_scaler_on_train, build_split_sequences
from backend.temporal.schema import STATE_FEATURE_NAMES
from backend.temporal.temporal_dataset import prepare_temporal_dataset


REQUIRED_NUMERIC_COLS = [
    "src_port", "dst_port", "protocol", "flow_duration", "flow_byts_s", "flow_pkts_s",
    "fwd_pkts_s", "bwd_pkts_s", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "fwd_header_len", "bwd_header_len", "fwd_seg_size_min", "fwd_act_data_pkts",
    "flow_iat_mean", "flow_iat_max", "flow_iat_min", "flow_iat_std",
    "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_mean", "fwd_iat_std",
    "bwd_iat_tot", "bwd_iat_max", "bwd_iat_min", "bwd_iat_mean", "bwd_iat_std",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt",
    "ece_flag_cnt", "down_up_ratio", "pkt_size_avg", "init_fwd_win_byts", "init_bwd_win_byts",
    "active_max", "active_min", "active_mean", "active_std",
    "idle_max", "idle_min", "idle_mean", "idle_std",
    "fwd_byts_b_avg", "fwd_pkts_b_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg",
    "fwd_blk_rate_avg", "bwd_blk_rate_avg", "fwd_seg_size_avg", "bwd_seg_size_avg",
    "cwr_flag_count", "subflow_fwd_pkts", "subflow_bwd_pkts", "subflow_fwd_byts", "subflow_bwd_byts",
]


def make_flow_csv(n_rows: int, start="2026-01-01 00:00:00", step_seconds=2, states=None) -> pd.DataFrame:
    """Builds a minimal-but-schema-complete synthetic flow table using the
    project's real cicflowmeter column names, spaced step_seconds apart."""
    ts = pd.date_range(start, periods=n_rows, freq=f"{step_seconds}s")
    data = {
        "src_ip": [f"10.0.0.{i % 5 + 1}" for i in range(n_rows)],
        "dst_ip": [f"10.0.1.{i % 3 + 1}" for i in range(n_rows)],
        "src_port": [40000 + i for i in range(n_rows)],
        "dst_port": [80] * n_rows,
        "protocol": [6] * n_rows,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for col in REQUIRED_NUMERIC_COLS:
        if col in ("src_port", "dst_port", "protocol"):
            continue
        data[col] = np.random.default_rng(42).uniform(0, 100, n_rows)
    df = pd.DataFrame(data)
    if states is None:
        states = ["BENIGN"] * n_rows
    df["Current_State"] = states
    return df


# ---------- 1. timestamp parsing ----------

def test_timestamp_parsing_valid():
    df = make_flow_csv(5)
    result = parse_timestamps(df["timestamp"])
    assert result.n_valid == 5
    assert result.n_invalid == 0


def test_timestamp_parsing_invalid_rows_reported_not_fabricated():
    df = make_flow_csv(5)
    df.loc[2, "timestamp"] = "not-a-timestamp"
    result = parse_timestamps(df["timestamp"])
    assert result.n_valid == 4
    assert result.n_invalid == 1
    assert 2 in result.invalid_row_indices
    assert pd.isna(result.parsed.iloc[2])  # never silently filled in


def test_timestamp_parsing_all_invalid_raises_at_orchestrator_level(tmp_path):
    df = make_flow_csv(3)
    df["timestamp"] = ["garbage"] * 3
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(TemporalError):
        prepare_temporal_dataset(csv_path, tmp_path / "out")


# ---------- 2. chronological sorting ----------

def test_chronological_sorting():
    df = make_flow_csv(5)
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    ts = parse_timestamps(shuffled["timestamp"]).parsed
    shuffled["timestamp_parsed"] = ts
    sorted_df = sort_chronologically(shuffled, "timestamp_parsed")
    assert sorted_df["timestamp_parsed"].is_monotonic_increasing


# ---------- 3. window assignment ----------

def test_window_assignment_buckets_correctly():
    df = make_flow_csv(10, step_seconds=3)  # spans 27s
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", window_size_seconds=10)
    assert (windowed["timestamp_parsed"] >= windowed["window_start"]).all()
    assert (windowed["timestamp_parsed"] < windowed["window_end"]).all()
    assert windowed["window_id"].nunique() >= 2  # 27s / 10s window spans 3 windows


# ---------- 4. aggregation ----------

def test_aggregation_sums_not_single_flow():
    df = make_flow_csv(4, step_seconds=1)
    df.loc[:, "tot_fwd_pkts"] = [10, 20, 30, 40]
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    assert len(states) == 1
    assert states.iloc[0]["flow_count"] == 4
    assert states.iloc[0]["total_fwd_packets"] == 100  # sum, not first/random row


# ---------- 5. dominant-state calculation ----------

def test_dominant_state_by_flow_count():
    df = make_flow_csv(5, step_seconds=1, states=["BENIGN", "BENIGN", "BENIGN", "DDoS", "DDoS"])
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    assert states.iloc[0]["dominant_state"] == "BENIGN"
    assert states.iloc[0]["benign_flow_count"] == 3
    assert states.iloc[0]["ddos_flow_count"] == 2


# ---------- 6. attack-present calculation ----------

def test_attack_present_flag():
    df = make_flow_csv(3, step_seconds=1, states=["BENIGN", "BENIGN", "PortScan"])
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    assert states.iloc[0]["dominant_state"] == "BENIGN"
    assert states.iloc[0]["attack_present"] == 1  # dominant class hides the attack; flag preserves it


def test_no_attack_present_when_all_benign():
    df = make_flow_csv(3, step_seconds=1, states=["BENIGN", "BENIGN", "BENIGN"])
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    assert states.iloc[0]["attack_present"] == 0


# ---------- 7. transition generation ----------

def test_transition_generation():
    df = make_flow_csv(20, step_seconds=5)  # spans 100s -> ~10 windows of 10s
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    transitions = build_state_transitions(states)
    assert len(transitions) == len(states) - 1
    assert (transitions["next_window"] - transitions["current_window"] > 0).all()


# ---------- 8. sequence generation (exact count) ----------

def test_sequence_count_matches_N_minus_L():
    df = make_flow_csv(30, step_seconds=5)  # ~150s -> ~15 windows
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    n_windows = len(states)
    seq = build_sequences(states, sequence_length=5)
    assert seq["X"].shape[0] == n_windows - 5
    assert seq["X"].shape == (n_windows - 5, 5, len(STATE_FEATURE_NAMES))


# ---------- 9. chronological train/val/test split ----------

def test_chronological_split_preserves_order():
    df = make_flow_csv(40, step_seconds=5)
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    train, val, test, meta = chronological_split(states)
    assert train["window_id"].max() < val["window_id"].min()
    assert val["window_id"].max() < test["window_id"].min()
    assert len(train) + len(val) + len(test) == len(states)


# ---------- 10. scaler fitted only on training data ----------

def test_scaler_fitted_only_on_train():
    df = make_flow_csv(40, step_seconds=5)
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    train, val, test, _ = chronological_split(states)
    scaler = fit_scaler_on_train(train)
    expected = train[STATE_FEATURE_NAMES].min().to_numpy()
    assert np.allclose(scaler.data_min_, expected)
    # scaler's fitted range must NOT reflect val/test-only extreme values
    if not val.empty:
        val_only_max = val[STATE_FEATURE_NAMES].to_numpy().max()
        train_max = train[STATE_FEATURE_NAMES].to_numpy().max()
        if val_only_max > train_max:
            assert scaler.data_max_.max() < val_only_max


# ---------- 11. insufficient-window handling ----------

def test_insufficient_windows_raises_clear_error():
    df = make_flow_csv(3, step_seconds=1)  # 1 window only
    df["timestamp_parsed"] = parse_timestamps(df["timestamp"]).parsed
    df = sort_chronologically(df, "timestamp_parsed")
    windowed = assign_windows(df, "timestamp_parsed", 10)
    states, _ = build_temporal_states(windowed, "Current_State", 10)
    assert len(states) == 1
    with pytest.raises(TemporalError, match="Insufficient temporal data"):
        build_sequences(states, sequence_length=5)


def test_insufficient_windows_end_to_end(tmp_path):
    df = make_flow_csv(3, step_seconds=1)
    csv_path = tmp_path / "short.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(TemporalError, match="Insufficient temporal data"):
        prepare_temporal_dataset(csv_path, tmp_path / "out", window_size_seconds=10, sequence_length=5)


# ---------- 12. invalid timestamp handling (end-to-end via orchestrator) ----------

def test_invalid_timestamps_excluded_not_fabricated(tmp_path):
    df = make_flow_csv(30, step_seconds=5)
    df.loc[[3, 7, 15], "timestamp"] = ["", "not-a-date", None]
    csv_path = tmp_path / "mixed.csv"
    df.to_csv(csv_path, index=False)
    summary = prepare_temporal_dataset(csv_path, tmp_path / "out", window_size_seconds=10, sequence_length=5)
    assert summary["timestamps_invalid"] == 3
    assert summary["timestamps_valid"] == 27


def test_missing_timestamp_column_raises(tmp_path):
    df = make_flow_csv(10)
    df = df.drop(columns=["timestamp"])
    csv_path = tmp_path / "no_ts.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(TemporalError, match="[Tt]imestamp"):
        prepare_temporal_dataset(csv_path, tmp_path / "out")


# ---------- end-to-end sanity on a real project-generated file, if present ----------

def test_end_to_end_on_real_generated_csv():
    real_csv = Path(r"D:\NIDS-Using-CICIDS2017-Dataset\features\capture_2026-08-28_232338_fa981afc.csv")
    if not real_csv.exists():
        pytest.skip("Real captured CSV not present in this environment")
    summary = prepare_temporal_dataset(
        real_csv, Path(r"C:\Users\Kaviya V\.claude\jobs\63713ff0\tmp\temporal_pytest_out"),
        window_size_seconds=10, sequence_length=5,
    )
    assert summary["total_windows"] > 0
    assert summary["train_sequences"] > 0
    assert set(summary["states_df"]["dominant_state"].unique()) <= {"BENIGN", "DDoS", "DoS", "PortScan"}
