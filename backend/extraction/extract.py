"""
CICFlowMeter feature extraction: .pcap -> flow-feature .csv.

The original Java CICFlowMeter (ahlashkari/CICFlowMeter) depends on
jnetpcap, a native library with no working 64-bit Windows/Java 23 build
path and no CLI without WinPcap dev headers. In its place this uses the
`cicflowmeter` PyPI package (v0.5.0, Scapy-based, MIT-licensed), which
implements the same flow-feature extraction logic without a native
dependency. See scripts/patch_cicflowmeter.py for the two upstream bugs
that had to be fixed for it to run at all, and README.md for setup.

No fabricated results: any failure (missing PCAP, empty PCAP, no flows
found, crash inside the sniffer) raises ExtractionError with the real
cause instead of returning a fake success.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


class ExtractionError(Exception):
    pass


def run_extraction(pcap_path: Path, features_dir: Path) -> dict:
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise ExtractionError(f"PCAP not found: {pcap_path}")
    if pcap_path.stat().st_size == 0:
        raise ExtractionError(f"PCAP is empty: {pcap_path}")

    features_dir.mkdir(parents=True, exist_ok=True)
    csv_path = features_dir / f"{pcap_path.stem}.csv"

    try:
        from cicflowmeter.sniffer import create_sniffer
    except ImportError as e:
        raise ExtractionError(
            f"cicflowmeter package not installed in this environment: {e}"
        )

    t0 = time.time()
    try:
        sniffer, session = create_sniffer(
            input_file=str(pcap_path),
            input_interface=None,
            output_mode="csv",
            output=str(csv_path),
            fields=None,
            verbose=False,
        )
        sniffer.start()
        sniffer.join()
    except Exception as e:
        raise ExtractionError(f"CICFlowMeter (cicflowmeter) extraction failed: {e}")
    finally:
        try:
            if hasattr(session, "_gc_stop"):
                session._gc_stop.set()
                session._gc_thread.join(timeout=2.0)
            session.flush_flows()
        except Exception:
            pass
    elapsed = round(time.time() - t0, 2)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise ExtractionError(
            "CICFlowMeter ran but produced no CSV output "
            "(no TCP/UDP flows found in the PCAP?)."
        )

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except pd.errors.EmptyDataError:
        raise ExtractionError(f"CICFlowMeter produced an empty CSV: {csv_path}")

    if df.empty:
        raise ExtractionError(f"CICFlowMeter CSV has no flow rows: {csv_path}")

    return {
        "input_pcap": str(pcap_path),
        "output_csv": str(csv_path),
        "flow_count": len(df),
        "feature_count": len(df.columns),
        "extraction_seconds": elapsed,
    }
