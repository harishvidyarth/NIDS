import json
from pathlib import Path

from backend.graph.analyzer import CommunicationGraphAnalyzer


def _flow(source: str, destination: str, service: str = "https", timestamp: float = 1.0) -> dict:
    return {
        "src_ip": source, "dst_ip": destination, "service": service,
        "bytes_out": 100, "bytes_in": 400, "timestamp": timestamp,
    }


def test_graph_build_aggregates_edges(tmp_path: Path):
    analyzer = CommunicationGraphAnalyzer(tmp_path / "baseline.json")
    result = analyzer.analyze(
        "one",
        conn_records=[_flow("10.0.0.2", "10.0.0.3", timestamp=2), _flow("10.0.0.2", "10.0.0.3")],
    )

    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    assert result["edges"][0]["flow_count"] == 2
    assert result["edges"][0]["bytes"] == 1000.0
    assert result["edges"][0]["first_seen"] == 1.0


def test_baseline_learning_then_lateral_movement_scores_high(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    analyzer = CommunicationGraphAnalyzer(baseline_path)
    benign = [
        _flow("10.0.0.2", "10.0.0.10"),
        _flow("10.0.0.3", "10.0.0.11"),
        _flow("10.0.0.4", "10.0.0.12"),
    ]
    learned = analyzer.analyze("baseline", conn_records=benign, mostly_benign=True)

    assert learned["baseline_learned"] is True
    assert baseline_path.exists()
    assert len(json.loads(baseline_path.read_text(encoding="utf-8"))["edges"]) == 3

    lateral = benign + [
        _flow("10.0.0.99", "10.0.0.10", "smb"),
        _flow("10.0.0.99", "10.0.0.11", "smb"),
        _flow("10.0.0.99", "10.0.0.12", "smb"),
        _flow("10.0.0.99", "10.0.0.13", "rdp"),
    ]
    result = analyzer.analyze("attack", conn_records=lateral)

    novel = [edge for edge in result["surprising_edges"] if edge["source"] == "10.0.0.99"]
    assert len(novel) == 4
    assert min(edge["edge_surprise"] for edge in novel) >= 0.6
    assert result["campaign_score"] >= 0.5


def test_deception_bump_is_bounded(tmp_path: Path):
    analyzer = CommunicationGraphAnalyzer(tmp_path / "baseline.json")
    analyzer.analyze("session", conn_records=[_flow("10.0.0.2", "10.0.0.3")])
    for _ in range(10):
        bumped = analyzer.bump_for_deception("session")

    assert bumped is not None
    assert bumped["campaign_score"] == 1.0
    assert bumped["deception_signal"] is True
