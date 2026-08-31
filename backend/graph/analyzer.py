"""Build and score a host communication graph without heavyweight ML.

The scorer intentionally uses stable, inspectable features.  It can later be
replaced by a trained graph model without changing the API response consumed by
the dashboard and triage service.
"""
from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from backend.ingest import get_ingest_store

DEFAULT_BASELINE_PATH = Path(__file__).resolve().parent / "benign_baseline.json"


def _pick(record: Any, *names: str) -> Any:
    if hasattr(record, "get"):
        for name in names:
            value = record.get(name)
            if value not in (None, "", "-"):
                return value
    return None


def _number(record: Any, *names: str) -> float:
    try:
        return float(_pick(record, *names) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _zscore(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation == 0:
        return 1.0 if value > mean else 0.0
    return max(0.0, (value - mean) / deviation)


def _bounded_z(value: float) -> float:
    return min(1.0, max(0.0, value / 3.0))


class CommunicationGraphAnalyzer:
    """Session graph builder with an optional persisted benign baseline."""

    def __init__(self, baseline_path: Path | None = None):
        self.baseline_path = baseline_path or DEFAULT_BASELINE_PATH
        self._lock = threading.RLock()
        self._latest: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _flow_records(flows: Any) -> list[Any]:
        if flows is None:
            return []
        if hasattr(flows, "to_dict"):
            return list(flows.to_dict(orient="records"))
        if isinstance(flows, dict):
            return [flows]
        return list(flows)

    def build(self, flows: Any = None, conn_records: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
        edge_data: dict[tuple[str, str], dict[str, Any]] = {}
        combined = self._flow_records(flows) + list(conn_records)
        for record in combined:
            source = _pick(record, "src_ip", "Src IP", "Source IP", "id.orig_h")
            destination = _pick(record, "dst_ip", "Dst IP", "Destination IP", "id.resp_h")
            if not source or not destination:
                continue
            key = (str(source), str(destination))
            edge = edge_data.setdefault(key, {
                "source": key[0], "target": key[1], "flow_count": 0,
                "bytes": 0.0, "services": set(), "first_seen": None,
            })
            edge["flow_count"] += 1
            edge["bytes"] += _number(
                record, "bytes", "totlen_fwd_pkts", "TotLen Fwd Pkts", "bytes_out", "orig_bytes",
            ) + _number(record, "totlen_bwd_pkts", "TotLen Bwd Pkts", "bytes_in", "resp_bytes")
            service = _pick(record, "service", "protocol", "proto", "Protocol")
            if service:
                edge["services"].add(str(service))
            timestamp = _number(record, "timestamp", "Timestamp", "ts")
            if timestamp and (edge["first_seen"] is None or timestamp < edge["first_seen"]):
                edge["first_seen"] = timestamp

        edges: list[dict[str, Any]] = []
        nodes: dict[str, dict[str, Any]] = {}
        for edge in edge_data.values():
            edge["services"] = sorted(edge["services"])
            edge["bytes"] = round(edge["bytes"], 3)
            edges.append(edge)
            nodes.setdefault(edge["source"], {"id": edge["source"], "in_degree": 0, "out_degree": 0})
            nodes.setdefault(edge["target"], {"id": edge["target"], "in_degree": 0, "out_degree": 0})
            nodes[edge["source"]]["out_degree"] += 1
            nodes[edge["target"]]["in_degree"] += 1
        return {
            "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
            "edges": sorted(edges, key=lambda edge: (edge["source"], edge["target"])),
        }

    def _load_baseline(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) and "edges" in data else None

    def _persist_baseline(self, graph: dict[str, Any]) -> None:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "edges": [
                {"source": edge["source"], "target": edge["target"], "services": edge["services"]}
                for edge in graph["edges"]
            ],
            "node_in_degrees": [node["in_degree"] for node in graph["nodes"]],
            "node_out_degrees": [node["out_degree"] for node in graph["nodes"]],
        }
        temporary = self.baseline_path.with_suffix(self.baseline_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.baseline_path)

    def analyze(
        self,
        session_id: str,
        flows: Any = None,
        conn_records: Iterable[dict[str, Any]] | None = None,
        mostly_benign: bool = False,
    ) -> dict[str, Any]:
        """Build and score a graph, learning only explicitly benign sessions."""
        if conn_records is None:
            conn_records = get_ingest_store().records(session_id, "conn")
        graph = self.build(flows, conn_records)
        with self._lock:
            baseline = self._load_baseline()
            learned = False
            if baseline is None and mostly_benign:
                self._persist_baseline(graph)
                baseline = self._load_baseline()
                learned = True

            baseline_edges = {
                (edge["source"], edge["target"]): set(edge.get("services", []))
                for edge in (baseline or {}).get("edges", [])
            }
            baseline_in = [float(value) for value in (baseline or {}).get("node_in_degrees", [])]
            baseline_out = [float(value) for value in (baseline or {}).get("node_out_degrees", [])]
            node_lookup = {node["id"]: node for node in graph["nodes"]}
            scored: list[dict[str, Any]] = []
            for edge in graph["edges"]:
                key = (edge["source"], edge["target"])
                prior_services = baseline_edges.get(key, set())
                unseen = 1.0 if key not in baseline_edges else 0.0
                new_service = 1.0 if set(edge["services"]) - prior_services else 0.0
                destination_degree = node_lookup[edge["target"]]["in_degree"]
                source_fanout = node_lookup[edge["source"]]["out_degree"]
                score = (
                    0.4 * unseen
                    + 0.2 * _bounded_z(_zscore(destination_degree, baseline_in))
                    + 0.2 * new_service
                    + 0.2 * _bounded_z(_zscore(source_fanout, baseline_out))
                )
                enriched = dict(edge)
                enriched["edge_surprise"] = round(min(1.0, score), 4)
                enriched["reasons"] = [
                    reason for active, reason in (
                        (unseen, "edge not present in benign baseline"),
                        (new_service, "new service on this host pair"),
                        (_zscore(destination_degree, baseline_in) > 0, "unusual destination in-degree"),
                        (_zscore(source_fanout, baseline_out) > 0, "unusual source fan-out"),
                    ) if active
                ]
                scored.append(enriched)

            scored.sort(key=lambda edge: (-edge["edge_surprise"], edge["source"], edge["target"]))
            surprising = [edge for edge in scored if edge["edge_surprise"] >= 0.5]
            top = scored[: min(5, len(scored))]
            campaign_score = sum(edge["edge_surprise"] for edge in top) / len(top) if top else 0.0
            result = {
                "session_id": str(session_id),
                "nodes": graph["nodes"],
                "edges": scored,
                "surprising_edges": surprising,
                "campaign_score": round(campaign_score, 4),
                "baseline_available": baseline is not None,
                "baseline_learned": learned,
                "scorer": "deterministic-edge-surprise-v1",
            }
            self._latest[str(session_id)] = result
            return result

    def latest(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            result = self._latest.get(str(session_id))
            return json.loads(json.dumps(result)) if result is not None else None

    def bump_for_deception(self, session_id: str, amount: float = 0.25) -> dict[str, Any] | None:
        """Raise an analyzed session's campaign score after a canary hit."""
        with self._lock:
            result = self._latest.get(str(session_id))
            if result is not None:
                result["campaign_score"] = round(min(1.0, result["campaign_score"] + amount), 4)
                result["deception_signal"] = True
            return json.loads(json.dumps(result)) if result is not None else None


_ANALYZER = CommunicationGraphAnalyzer()


def get_graph_analyzer() -> CommunicationGraphAnalyzer:
    return _ANALYZER
