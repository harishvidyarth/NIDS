"""
Parallel .pcap -> flow-feature .csv extraction.

`backend/extraction/extract.py:run_extraction` drives a single Scapy
`AsyncSniffer` replay of the whole capture on one GIL-bound thread, so a
large upload (~100k packets) spends minutes in one core. This module keeps
that function untouched as the unit of work and, for captures above a
packet-count threshold, splits the pcap into fixed-size chunks with
`editcap -c`, runs one `run_extraction` per chunk in a separate process
(escaping the GIL), and concatenates the per-chunk CSVs into the single
`features/<stem>.csv` the rest of the pipeline expects.

Trade-off: `editcap -c` cuts on packet boundaries, so a flow whose packets
straddle a cut is counted once per chunk. The inflation scales with the
number of cuts times how many flows are live at each cut, so keep
`chunk_size` large (default 20000) to minimise it. The return dict flags
`parallel`/`chunks` so callers can see a parallel run happened.

Any failure (missing binary, a chunk crashing, no CSV produced) raises
`ExtractionError` with the real cause — never a fabricated success.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from .extract import ExtractionError, run_extraction

DEFAULT_CHUNK_SIZE = 20_000
DEFAULT_THRESHOLD = 20_000


def _editcap_bin() -> str:
    found = shutil.which("editcap")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/editcap", "/usr/bin/editcap", "/usr/local/bin/editcap"):
        if Path(candidate).is_file():
            return candidate
    raise ExtractionError("editcap not found on PATH — cannot split the capture for parallel extraction.")


def _resolve_packet_count(pcap_path: Path, packet_count: int | None) -> int | None:
    if packet_count is not None:
        return packet_count
    try:
        from ..capture.capture import _capfile_stats

        return _capfile_stats(pcap_path).get("packet_count")
    except Exception:
        return None


def _extract_chunk(job: tuple[str, str]) -> dict:
    """Worker: one chunk pcap -> its own features dir. Module-level and
    closure-free so it survives `spawn` pickling."""
    pcap_str, features_dir_str = job
    return run_extraction(Path(pcap_str), Path(features_dir_str))


def run_parallel_extraction(
    pcap_path: Path,
    features_dir: Path,
    *,
    packet_count: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    threshold: int = DEFAULT_THRESHOLD,
    max_workers: int | None = None,
) -> dict:
    """Same contract as `run_extraction` (returns input_pcap / output_csv /
    flow_count / feature_count / extraction_seconds), with `parallel` and
    `chunks` added. Falls back to a plain `run_extraction` for small
    captures or when the packet count can't be determined."""
    pcap_path = Path(pcap_path)
    features_dir = Path(features_dir)
    if not pcap_path.exists():
        raise ExtractionError(f"PCAP not found: {pcap_path}")

    n_packets = _resolve_packet_count(pcap_path, packet_count)
    if n_packets is None or n_packets < threshold:
        result = run_extraction(pcap_path, features_dir)
        result.setdefault("parallel", False)
        result.setdefault("chunks", 1)
        return result

    features_dir.mkdir(parents=True, exist_ok=True)
    final_csv = features_dir / f"{pcap_path.stem}.csv"
    editcap = _editcap_bin()
    tmp_root = Path(tempfile.mkdtemp(prefix="nids_pex_"))
    t0 = time.time()
    try:
        split_prefix = tmp_root / "chunk.pcap"
        proc = subprocess.run(
            [editcap, "-c", str(chunk_size), str(pcap_path), str(split_prefix)],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise ExtractionError(
                f"editcap failed to split the capture: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        chunk_pcaps = sorted(tmp_root.glob("chunk_*.pcap"))
        if not chunk_pcaps:
            raise ExtractionError("editcap produced no chunks — is the capture readable?")

        jobs: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunk_pcaps):
            chunk_feat_dir = tmp_root / "feat" / str(index)
            chunk_feat_dir.mkdir(parents=True, exist_ok=True)
            jobs.append((str(chunk), str(chunk_feat_dir)))

        cpu = os.cpu_count() or 2
        workers = max_workers or max(1, cpu - 1)
        workers = min(workers, len(jobs))

        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_extract_chunk, job): i for i, job in enumerate(jobs)}
            try:
                for future in futures:
                    results.append(future.result())
            except Exception as error:  # one chunk failed -> abort the whole run
                for pending in futures:
                    pending.cancel()
                failed_index = futures.get(future, "?")
                raise ExtractionError(
                    f"parallel extraction failed on chunk {failed_index}: {error}"
                ) from error

        frames = []
        for chunk_result in results:
            chunk_csv = Path(chunk_result["output_csv"])
            if chunk_csv.is_file() and chunk_csv.stat().st_size > 0:
                frames.append(pd.read_csv(chunk_csv, low_memory=False))
        if not frames:
            raise ExtractionError(
                "Parallel extraction produced no flow rows across any chunk "
                "(no TCP/UDP flows in the PCAP?)."
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.reindex(columns=list(frames[0].columns))
        combined.to_csv(final_csv, index=False)
        elapsed = round(time.time() - t0, 2)

        return {
            "input_pcap": str(pcap_path),
            "output_csv": str(final_csv),
            "flow_count": int(len(combined)),
            "feature_count": int(len(combined.columns)),
            "extraction_seconds": elapsed,
            "parallel": True,
            "chunks": len(chunk_pcaps),
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
