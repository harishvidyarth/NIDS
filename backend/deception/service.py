"""Thread-safe in-memory storage for XDR deception canary hits."""
from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


HONEYTOKEN_PATH = Path(__file__).with_name("honeytoken_credentials.txt")
HONEYTOKEN_DISPLAY_PATH = "backend/deception/honeytoken_credentials.txt"
MAX_HITS = 1_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CanaryStore:
    """Record canary accesses and expose normalized high-confidence events."""

    def __init__(self, clock: Callable[[], datetime] | None = None):
        self._clock = clock or _utc_now
        self._lock = threading.RLock()
        self._hits: list[dict] = []

    @property
    def honeytoken_path(self) -> str:
        return HONEYTOKEN_DISPLAY_PATH

    def record_hit(self, source_ip: str | None, user_agent: str | None) -> dict:
        timestamp = self._clock()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        hit = {
            "hit_id": str(uuid.uuid4()),
            "source_ip": (source_ip or "unknown")[:128],
            "user_agent": (user_agent or "unknown")[:512],
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            "honeytoken_path": self.honeytoken_path,
        }
        with self._lock:
            self._hits.append(hit)
            if len(self._hits) > MAX_HITS:
                del self._hits[:-MAX_HITS]
        return dict(hit)

    def list_hits(self) -> list[dict]:
        with self._lock:
            return [dict(hit) for hit in self._hits]

    def high_confidence_events(self) -> list[dict]:
        return [self.as_event(hit) for hit in self.list_hits()]

    @staticmethod
    def as_event(hit: dict) -> dict:
        source = str(hit.get("source_ip", "unknown"))
        return {
            "event_id": str(hit.get("hit_id", "unknown")),
            "event_type": "DECEPTION_CANARY_HIT",
            "severity": "HIGH",
            "confidence": 1.0,
            "timestamp": str(hit.get("timestamp", "")),
            "source_ip": source,
            "evidence": f"Deception canary accessed from {source}.",
        }

    def campaign_score_boost(self) -> float:
        """Return a bounded boost for graph/triage campaign scoring."""
        with self._lock:
            return min(0.35, 0.20 * len(self._hits))

    def clear(self) -> None:
        """Reset process-local demo state. Primarily useful for tests."""
        with self._lock:
            self._hits.clear()


default_canary_store = CanaryStore()
