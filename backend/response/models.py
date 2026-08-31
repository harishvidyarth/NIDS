from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResponseTarget:
    source_ip: str
    victim_ip: str | None = None
    protocol: str | None = None
    destination_port: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyDecision:
    executable: bool
    attack_class: str | None
    confidence_source: str
    targets: list[ResponseTarget] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    upstream_recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value
