"""Normalize Zeek JSON logs and derive lightweight session enrichment.

The store is deliberately in-memory: raw Zeek telemetry is demo input, while
the existing pipeline remains the source of durable capture and prediction
artifacts.  Callers select the session key used to join the enrichment later.
"""
from __future__ import annotations

import json
import math
import statistics
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_JA3_BASELINE = _MODULE_DIR / "benign_ja3.json"
MAX_LOG_BYTES = 10 * 1024 * 1024
MAX_RECORDS = 100_000


def _text(record: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = record.get(name)
        if value not in (None, "", "-"):
            return str(value)
    return None


def _number(record: dict[str, Any], *names: str) -> float:
    value = _text(record, *names)
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def query_entropy(query: str) -> float:
    """Return Shannon entropy in bits per character for one DNS name."""
    if not query:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for character in query.lower():
        counts[character] += 1
    length = len(query)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _log_type(record: dict[str, Any]) -> str | None:
    explicit = _text(record, "_path", "log_type", "type")
    if explicit:
        value = explicit.lower().removesuffix(".log")
        if value in {"conn", "dns", "ssl"}:
            return value
    if "query" in record or "rcode_name" in record:
        return "dns"
    if any(key in record for key in ("server_name", "ja3", "ja4")):
        return "ssl"
    if "id.orig_h" in record and "id.resp_h" in record:
        return "conn"
    return None


def normalize_record(raw: dict[str, Any] | str) -> dict[str, Any]:
    """Normalize supported Zeek JSON fields; reject malformed/unknown rows."""
    if isinstance(raw, str):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Zeek log line is not valid JSON") from exc
    elif isinstance(raw, dict):
        record = dict(raw)
    else:
        raise ValueError("Zeek log line must contain one JSON object")
    if not isinstance(record, dict):
        raise ValueError("Zeek log line must contain one JSON object")
    kind = _log_type(record)
    if kind is None:
        raise ValueError("Unsupported Zeek log type; expected conn, dns, or ssl")

    normalized: dict[str, Any] = {
        "log_type": kind,
        "timestamp": _number(record, "ts", "timestamp"),
        "uid": _text(record, "uid"),
        "src_ip": _text(record, "id.orig_h", "src_ip"),
        "src_port": int(_number(record, "id.orig_p", "src_port")),
        "dst_ip": _text(record, "id.resp_h", "dst_ip"),
        "dst_port": int(_number(record, "id.resp_p", "dst_port")),
    }
    if kind == "conn":
        normalized.update({
            "protocol": _text(record, "proto", "protocol"),
            "service": _text(record, "service"),
            "bytes_out": _number(record, "orig_bytes", "bytes_out"),
            "bytes_in": _number(record, "resp_bytes", "bytes_in"),
        })
    elif kind == "dns":
        normalized.update({
            "query": _text(record, "query") or "",
            "rcode": _text(record, "rcode_name", "rcode") or "",
        })
    else:
        normalized.update({
            "sni": _text(record, "server_name", "sni"),
            "ja3": _text(record, "ja3"),
            "ja4": _text(record, "ja4"),
        })
    return normalized


class ZeekIngestStore:
    """Thread-safe normalized record store keyed by capture/session id."""

    def __init__(self, ja3_baseline_path: Path | None = None):
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self.ja3_baseline_path = ja3_baseline_path or DEFAULT_JA3_BASELINE

    def ingest(self, session_id: str, logs: Iterable[dict[str, Any] | str]) -> dict[str, Any]:
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id is required")
        rows = [normalize_record(row) for row in logs]
        if len(rows) > MAX_RECORDS:
            raise ValueError(f"At most {MAX_RECORDS} Zeek records may be ingested at once")
        with self._lock:
            self._sessions.setdefault(str(session_id), []).extend(rows)
        return {
            "session_id": str(session_id),
            "accepted": len(rows),
            "record_count": len(self.records(str(session_id))),
            "enrichment": self.enrichment(str(session_id)),
        }

    def ingest_directory(self, session_id: str, directory: str | Path) -> dict[str, Any]:
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Zeek path must be a directory")
        rows: list[dict[str, Any] | str] = []
        files: list[str] = []
        for candidate in sorted(root.glob("*.log")):
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError("Zeek log symlink escapes the supplied directory") from exc
            if not resolved.is_file():
                continue
            if resolved.stat().st_size > MAX_LOG_BYTES:
                raise ValueError(f"Zeek log exceeds {MAX_LOG_BYTES} bytes: {candidate.name}")
            files.append(candidate.name)
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip() and not line.lstrip().startswith("#"):
                        rows.append(line)
                    if len(rows) > MAX_RECORDS:
                        raise ValueError(f"At most {MAX_RECORDS} Zeek records may be ingested at once")
        result = self.ingest(session_id, rows)
        result["files"] = files
        return result

    def records(self, session_id: str, log_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = [dict(row) for row in self._sessions.get(str(session_id), [])]
        if log_type is not None:
            rows = [row for row in rows if row["log_type"] == log_type]
        return rows

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(str(session_id), None)

    def _benign_ja3(self) -> set[str]:
        try:
            data = json.loads(self.ja3_baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        values = data.get("ja3", []) if isinstance(data, dict) else data
        return {str(value) for value in values if value}

    def enrichment(self, session_id: str) -> dict[str, Any]:
        rows = self.records(session_id)
        dns = [row for row in rows if row["log_type"] == "dns"]
        ssl = [row for row in rows if row["log_type"] == "ssl"]
        conn = [row for row in rows if row["log_type"] == "conn"]

        entropies = [query_entropy(row["query"]) for row in dns if row.get("query")]
        nxdomain = sum(str(row.get("rcode", "")).upper() in {"NXDOMAIN", "3"} for row in dns)
        ja3_values = [str(row["ja3"]) for row in ssl if row.get("ja3")]
        ja4_values = [str(row["ja4"]) for row in ssl if row.get("ja4")]
        baseline = self._benign_ja3()
        novel = sum(value not in baseline for value in ja3_values)

        pair_times: dict[tuple[str, str], list[float]] = defaultdict(list)
        asymmetry: list[float] = []
        for row in conn:
            if row.get("src_ip") and row.get("dst_ip"):
                pair_times[(row["src_ip"], row["dst_ip"])].append(float(row["timestamp"]))
            outbound = float(row.get("bytes_out", 0.0))
            inbound = float(row.get("bytes_in", 0.0))
            total = outbound + inbound
            if total > 0:
                asymmetry.append(abs(outbound - inbound) / total)

        beacon_scores: list[float] = []
        for timestamps in pair_times.values():
            ordered = sorted(timestamps)
            intervals = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
            if len(intervals) < 2:
                continue
            mean = statistics.fmean(intervals)
            if mean > 0:
                beacon_scores.append(max(0.0, min(1.0, 1.0 - statistics.pstdev(intervals) / mean)))

        return {
            "dns_query_entropy_mean": statistics.fmean(entropies) if entropies else 0.0,
            "unique_sni_count": len({row["sni"] for row in ssl if row.get("sni")}),
            "nxdomain_ratio": nxdomain / len(dns) if dns else 0.0,
            "ja3": sorted(set(ja3_values)),
            "ja4": sorted(set(ja4_values)),
            "ja3_novelty": novel / len(ja3_values) if ja3_values else 0.0,
            "beacon_score_max": max(beacon_scores, default=0.0),
            "byte_asymmetry_max": max(asymmetry, default=0.0),
            "record_count": len(rows),
        }


_STORE = ZeekIngestStore()


def get_ingest_store() -> ZeekIngestStore:
    return _STORE
