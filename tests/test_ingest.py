import json
from pathlib import Path

import pytest

from backend.ingest.store import ZeekIngestStore, query_entropy


def _conn(timestamp: float, source: str = "10.0.0.2", destination: str = "198.51.100.8") -> dict:
    return {
        "_path": "conn", "ts": timestamp, "id.orig_h": source, "id.resp_h": destination,
        "orig_bytes": 100, "resp_bytes": 0, "service": "ssl",
    }


def test_query_entropy_distinguishes_repetition_from_variety():
    assert query_entropy("aaaaaaaaaaaa") == 0.0
    assert query_entropy("a9f2k8m4.example") > 3.0


def test_beacon_novelty_and_asymmetry(tmp_path: Path):
    baseline = tmp_path / "ja3.json"
    baseline.write_text(json.dumps({"ja3": ["known"]}), encoding="utf-8")
    store = ZeekIngestStore(baseline)
    logs = [_conn(100.0), _conn(110.0), _conn(120.0), _conn(130.0)]
    logs.extend([
        {"_path": "ssl", "ts": 101, "server_name": "one.test", "ja3": "known", "ja4": "v1"},
        {"_path": "ssl", "ts": 102, "server_name": "two.test", "ja3": "novel", "ja4": "v2"},
        {"_path": "dns", "ts": 103, "query": "a9f2k8m4.example", "rcode_name": "NXDOMAIN"},
    ])
    result = store.ingest("capture-1", logs)
    enrichment = result["enrichment"]

    assert enrichment["beacon_score_max"] == pytest.approx(1.0)
    assert enrichment["byte_asymmetry_max"] == pytest.approx(1.0)
    assert enrichment["ja3_novelty"] == pytest.approx(0.5)
    assert enrichment["ja3"] == ["known", "novel"]
    assert enrichment["unique_sni_count"] == 2
    assert enrichment["nxdomain_ratio"] == pytest.approx(1.0)


def test_directory_round_trip_and_sample_has_200_records(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    records = [
        {"_path": "conn", "ts": 1, "id.orig_h": "10.0.0.2", "id.resp_h": "192.0.2.2"},
        {"_path": "dns", "ts": 2, "query": "example.test", "rcode_name": "NOERROR"},
    ]
    (log_dir / "conn.log").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    store = ZeekIngestStore(tmp_path / "missing-baseline.json")

    result = store.ingest_directory("session", log_dir)

    assert result["accepted"] == 2
    assert result["record_count"] == 2
    assert result["files"] == ["conn.log"]
    assert [row["log_type"] for row in store.records("session")] == ["conn", "dns"]

    sample = Path("backend/ingest/sample_zeek/sample.log")
    assert len(sample.read_text(encoding="utf-8").splitlines()) == 200


def test_unknown_log_type_is_rejected():
    store = ZeekIngestStore()
    with pytest.raises(ValueError, match="Unsupported Zeek log type"):
        store.ingest("session", [{"message": "not zeek telemetry"}])
    with pytest.raises(ValueError, match="one JSON object"):
        store.ingest("session", [["not", "an", "object"]])
