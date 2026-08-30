"""Output helpers for the `nids` CLI — one JSON path, one compact human path."""
from __future__ import annotations

import json
from typing import Any


class CliError(Exception):
    """A handled CLI failure. `code` is the process exit code (3 = a needed
    artifact / dataset is missing, 4 = a runtime failure)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _scalar(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _human(obj: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.extend(_human(value, indent + 1))
            else:
                shown = "{}" if isinstance(value, (dict, list)) else _scalar(value)
                lines.append(f"{pad}{key}: {shown}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}- [{i}]")
                lines.extend(_human(value, indent + 1))
            else:
                lines.append(f"{pad}- {_scalar(value)}")
    else:
        lines.append(f"{pad}{_scalar(obj)}")
    return lines


def emit(payload: Any, as_json: bool) -> None:
    """Print a handler's result. `--json` -> pretty JSON; otherwise an
    indented key/value view. `None` prints nothing."""
    if payload is None:
        return
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    print("\n".join(_human(payload)))
