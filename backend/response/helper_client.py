from __future__ import annotations

from pathlib import Path
from typing import Any


ALLOWED_OPERATIONS = {"apply_plan", "verify_action", "rollback_action"}


class PrivilegedHelperClient:
    """Fixed-argv JSON client for an independently installed privileged helper."""

    def __init__(self, executable: str | Path):
        path = Path(executable)
        if not path.is_absolute():
            raise ValueError("Privileged helper path must be absolute.")
        self.executable = str(path)

    def __call__(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation not in ALLOWED_OPERATIONS:
            raise ValueError("Unsupported privileged helper operation.")
        raise RuntimeError("Privileged firewall execution is disabled in the XDR prototype.")


def configured_helper() -> PrivilegedHelperClient | None:
    return None
