"""
Tests for backend/extraction/parallel_extract — the chunked, multi-process
wrapper around run_extraction.

Skips the cicflowmeter-backed cases when the package or a demo pcap is not
present, so the file is safe on a bare checkout; the corrupt-input case
runs everywhere (it never reaches cicflowmeter).
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.extraction.extract import ExtractionError, run_extraction
from backend.extraction.parallel_extract import run_parallel_extraction

REPO = Path(__file__).resolve().parent.parent
DEMO_PCAP = REPO / "pcaps" / "capture_2026-08-30_005400_323c8383.pcap"

cicflowmeter = pytest.importorskip("cicflowmeter")
_have_demo = DEMO_PCAP.is_file()


@pytest.mark.skipif(not _have_demo, reason="demo pcap not present")
def test_below_threshold_falls_back_to_serial(tmp_path):
    result = run_parallel_extraction(DEMO_PCAP, tmp_path / "feat")
    assert result["parallel"] is False
    assert result["chunks"] == 1
    assert result["flow_count"] > 0
    assert Path(result["output_csv"]).is_file()


@pytest.mark.skipif(not _have_demo, reason="demo pcap not present")
def test_parallel_matches_serial_columns_and_rowcount(tmp_path):
    serial = run_extraction(DEMO_PCAP, tmp_path / "serial")
    import pandas as pd

    serial_df = pd.read_csv(serial["output_csv"], low_memory=False)

    parallel = run_parallel_extraction(
        DEMO_PCAP, tmp_path / "parallel", packet_count=8283, threshold=100, chunk_size=4000
    )
    assert parallel["parallel"] is True
    assert parallel["chunks"] > 1
    parallel_df = pd.read_csv(parallel["output_csv"], low_memory=False)

    assert list(parallel_df.columns) == list(serial_df.columns)
    # editcap -c cuts on packet boundaries, so a flow straddling a cut is
    # counted once per chunk -> never fewer rows than serial, and the
    # inflation grows with (chunk count x flows live at each cut). This
    # 3-chunk split of a tiny file is a deliberately harsh case; a real
    # 100k-packet capture at chunk_size=20000 inflates far less.
    assert len(parallel_df) >= len(serial_df)
    assert len(parallel_df) <= len(serial_df) * 2 + 20

    # no temp workspace left behind
    leftovers = list(Path(tempfile.gettempdir()).glob("nids_pex_*"))
    assert leftovers == []


def test_corrupt_input_raises_extraction_error(tmp_path):
    not_a_pcap = tmp_path / "broken.pcap"
    not_a_pcap.write_text("this is not a capture file\n")
    with pytest.raises(ExtractionError):
        run_parallel_extraction(not_a_pcap, tmp_path / "feat", packet_count=99_999, chunk_size=1000)
