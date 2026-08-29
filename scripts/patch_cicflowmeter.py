"""
Post-install fix for the `cicflowmeter` PyPI package (v0.5.0), which we use
in place of the original Java CICFlowMeter (that tool needs jnetpcap, a
native library with no working 64-bit/Java 23 Windows build path).

The package as published on PyPI has two bugs that break every capture/file
invocation:
  1. `create_sniffer()`'s CLI callers pass positional args in an order that
     doesn't match its signature, so `verbose` (a bool) lands in the
     `fields` parameter and crashes on `fields.split(",")`.
  2. Every `AsyncSniffer(...)` call passes `filter="ip and (tcp or udp)"`,
     which makes Scapy shell out to a real `tcpdump` binary to compile the
     BPF filter. That's not present on a Wireshark-only Windows box (tshark/
     dumpcap are not drop-in tcpdump replacements). `FlowSession.process()`
     already filters to TCP/UDP itself, so the BPF filter is redundant.

Usage:  python scripts/patch_cicflowmeter.py <path-to-venv-python.exe>
Idempotent — safe to run repeatedly (e.g. every fresh `pip install -r
requirements.txt`), and only touches this one file inside cicflowmeter.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_sniffer_path(python_exe: str) -> Path:
    proc = subprocess.run(
        [python_exe, "-c", "import cicflowmeter, os; print(os.path.dirname(cicflowmeter.__file__))"],
        capture_output=True, text=True, check=True,
    )
    return Path(proc.stdout.strip()) / "sniffer.py"


FIX_CALL_ORDER = (
    (
        "def create_sniffer(\n"
        "    input_file, input_interface, output_mode, output, input_directory=None, fields=None, verbose=False\n"
        "):",
    ),
)


def patch(sniffer_path: Path) -> bool:
    text = sniffer_path.read_text()
    original = text
    changed = False

    if "def create_sniffer(\n    input_file, input_interface, output_mode, output, input_directory=None, fields=None, verbose=False\n):" not in text:
        # Older/unpatched signature: positional (input_file, input_interface,
        # output_mode, output_file, verbose, fields). Normalize to keyword-safe order.
        import re
        text, n = re.subn(
            r"def create_sniffer\([^)]*\):",
            "def create_sniffer(\n"
            "    input_file, input_interface, output_mode, output, input_directory=None, fields=None, verbose=False\n"
            "):",
            text, count=1,
        )
        changed = changed or n > 0

    if 'filter="ip and (tcp or udp)"' in text or "filter='ip and (tcp or udp)'" in text:
        text = text.replace(',\n            filter="ip and (tcp or udp)"', "")
        text = text.replace(",\n            filter='ip and (tcp or udp)'", "")
        text = text.replace(', filter="ip and (tcp or udp)"', "")
        text = text.replace(", filter='ip and (tcp or udp)'", "")
        changed = True

    if changed and text != original:
        sniffer_path.write_text(text)
    return changed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python patch_cicflowmeter.py <path-to-venv-python.exe>")
        sys.exit(1)
    sniffer_path = find_sniffer_path(sys.argv[1])
    if not sniffer_path.exists():
        print(f"cicflowmeter sniffer.py not found at {sniffer_path} — is it installed?")
        sys.exit(1)
    did_change = patch(sniffer_path)
    print(f"{'Patched' if did_change else 'Already OK, no change needed:'} {sniffer_path}")
