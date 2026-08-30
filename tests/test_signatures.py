"""Deterministic cross-flow signature layer (backend/prediction/signatures.py)."""
from __future__ import annotations

import pandas as pd

from backend.prediction.signatures import flow_signatures, _port_access_pattern


def _benign_rows(n=20):
    return [
        dict(src_ip="10.1.1.2", dst_ip=f"172.16.0.{i}", src_port=5000 + i,
             dst_port=443, flow_pkts_s=2.0, syn_flag_cnt=1, ack_flag_cnt=5,
             totlen_fwd_pkts=1500)
        for i in range(n)
    ]


def test_ddos_source_fanin_flagged_even_with_benign_majority():
    rows = [
        dict(timestamp="2026-08-30 00:00:00", src_ip=f"10.0.0.{i}", dst_ip="192.168.1.5", src_port=40000 + i,
             dst_port=80, flow_pkts_s=50.0, syn_flag_cnt=1, ack_flag_cnt=1,
             tot_fwd_pkts=5, tot_bwd_pkts=0, totlen_fwd_pkts=200)
        for i in range(60)
    ] + _benign_rows(20)
    r = flow_signatures(pd.DataFrame(rows))
    assert r["attack_class"] == "DDoS"
    assert r["counts"]["DDoS"] == 60
    assert any(h["rule"] == "rolling-source-fanin-flood" for h in r["hits"])


def test_syn_flood_flagged_as_dos():
    rows = [
        dict(src_ip="10.5.5.5", dst_ip="192.168.1.7", src_port=7000 + i,
             dst_port=80, flow_pkts_s=500.0, syn_flag_cnt=40, ack_flag_cnt=1,
             totlen_fwd_pkts=0)
        for i in range(5)
    ]
    r = flow_signatures(pd.DataFrame(rows))
    assert r["attack_class"] == "DoS"


def test_sequential_port_sweep_flagged_as_portscan():
    rows = [
        dict(src_ip="10.9.9.9", dst_ip="192.168.1.9", src_port=6000,
             dst_port=1000 + p, flow_pkts_s=5.0, syn_flag_cnt=1, ack_flag_cnt=0,
             totlen_fwd_pkts=60)
        for p in range(50)
    ]
    r = flow_signatures(pd.DataFrame(rows))
    assert r["attack_class"] == "PortScan"
    assert r["port_scan"]["pattern"] == "sequential"
    assert r["port_scan"]["unique_ports"] == 50


def test_all_benign_is_benign():
    r = flow_signatures(pd.DataFrame(_benign_rows(30)))
    assert r["attack_class"] is None
    assert r["counts"] == {"BENIGN": 30}


def test_port_access_pattern_classes():
    assert _port_access_pattern([20, 21, 22, 23, 24]) == "sequential"
    assert _port_access_pattern([100, 4000, 22, 8080, 53, 3389]) == "randomised"
    assert _port_access_pattern([10, 20, 30, 40]) == "strided"
    assert _port_access_pattern([80]) == "none"


def test_missing_columns_degrades_to_benign():
    r = flow_signatures(pd.DataFrame({"flow_pkts_s": [1, 2, 3]}))
    assert r["attack_class"] is None
    assert r["states"] == ["BENIGN", "BENIGN", "BENIGN"]


def test_incidental_source_fanin_across_capture_is_not_ddos():
    rows = []
    for index in range(25):
        rows.append({
            "timestamp": f"2026-08-30 00:{index:02d}:00",
            "src_ip": f"10.0.0.{index + 1}", "dst_ip": "10.0.1.10", "dst_port": 443,
            "flow_pkts_s": 5, "tot_fwd_pkts": 5, "tot_bwd_pkts": 0,
            "syn_flag_cnt": 1, "ack_flag_cnt": 1, "totlen_fwd_pkts": 200,
        })
    result = flow_signatures(pd.DataFrame(rows))
    assert result["attack_class"] is None
    assert not any(hit["rule"] == "rolling-source-fanin-flood" for hit in result["hits"])


def test_distributed_flood_requires_concurrent_sources_and_victim_concentration():
    rows = [{
        "timestamp": "2026-08-30 00:00:00", "src_ip": f"10.0.0.{index + 1}",
        "dst_ip": "10.0.1.10", "dst_port": 443, "flow_pkts_s": 500,
        "tot_fwd_pkts": 8, "tot_bwd_pkts": 0, "syn_flag_cnt": 1,
        "ack_flag_cnt": 0, "totlen_fwd_pkts": 80,
    } for index in range(25)]
    result = flow_signatures(pd.DataFrame(rows))
    hit = next(hit for hit in result["hits"] if hit["rule"] == "rolling-source-fanin-flood")
    assert result["attack_class"] == "DDoS"
    assert hit["evidence"]["unique_sources"] == 25
    assert hit["evidence"]["victim_flow_ratio"] == 1.0
    assert hit["evidence"]["packet_rate"] == 200.0


def test_single_source_flood_is_dos_not_ddos():
    rows = [{
        "timestamp": "2026-08-30 00:00:00", "src_ip": "10.0.0.1",
        "dst_ip": "10.0.1.10", "dst_port": 443, "flow_pkts_s": 1000,
        "tot_fwd_pkts": 200, "tot_bwd_pkts": 0, "syn_flag_cnt": 200,
        "ack_flag_cnt": 0, "totlen_fwd_pkts": 1200,
    }]
    result = flow_signatures(pd.DataFrame(rows))
    assert result["attack_class"] == "DoS"
    assert all(hit["state"] != "DDoS" for hit in result["hits"])
