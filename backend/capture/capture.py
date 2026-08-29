"""
Live packet capture, wrapping Wireshark's `dumpcap`/`capinfos` on Windows
and `tcpdump` on Linux. No fabricated packet counts or durations: packet
count and final duration are read back from the real capture file via
`capinfos` (Windows) or `capinfos`/`tshark` if present, falling back to
None ("not available") rather than a guess.

This does not reuse scripts/packetsniff.sh directly: that script requires
explicit source/destination IP filters and has no stop/status/duration/
packet-count reporting, so it cannot satisfy the "select interface, start,
stop, show status/duration/count" requirement on its own. It stays in
scripts/ for manual/reference use; this module is the programmatic
equivalent the API needs.
"""
from __future__ import annotations

import platform
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

IS_WINDOWS = platform.system() == "Windows"

WIRESHARK_DIR = Path(r"C:\Program Files\Wireshark")
DUMPCAP = WIRESHARK_DIR / "dumpcap.exe"
CAPINFOS = WIRESHARK_DIR / "capinfos.exe"
TSHARK = WIRESHARK_DIR / "tshark.exe"

# Live-capture target: distinct from, and unrelated to, the packet-table
# PAGE size (PACKETS_PAGE_SIZE-equivalent lives in the API layer / frontend
# and only bounds how many rows are read back for display — it never
# limits what dumpcap actually captures). This bounds the capture itself:
# dumpcap/tcpdump are told to stop once this many packets have been
# written, via their own native `-c` flag, so the capture process exits
# on its own rather than needing any Python-side packet counting.
#
# Both a packet-count target AND a duration window are on by default:
# whichever condition is met first stops the capture (dumpcap/tcpdump's
# own -c and -a duration:N autostop flags). packet_target is set to
# 10000 so a capture can actually accumulate up to that many packets
# instead of being cut short early; duration is widened to 300s (rather
# than the old 120s) so there is realistically enough elapsed time for
# 10000 packets to arrive on a normal interface before duration cuts it
# off first — see backend/temporal for why real elapsed time still
# matters (10-second windows) even though packet count is now the
# primary target callers care about.
DEFAULT_CAPTURE_DURATION_SECONDS = 300
DEFAULT_CAPTURE_PACKET_TARGET = 10000
# Hard safety net so a capture on a near-idle interface can't run forever
# waiting to reach its target — enforced by the API layer polling
# check_and_finalize() on the same cadence it already polls capture
# status, not by a separate watchdog thread. Comfortably above the
# default duration target so it only ever fires as a genuine backstop.
CAPTURE_SAFETY_TIMEOUT_SECONDS = 600

_PACKET_FIELDS = [
    "frame.number", "frame.time_relative", "ip.src", "ipv6.src", "ip.dst",
    "ipv6.dst", "_ws.col.Protocol", "tcp.srcport", "udp.srcport",
    "tcp.dstport", "udp.dstport", "frame.len", "_ws.col.Info",
]


class CaptureError(Exception):
    pass


def list_interfaces() -> list[dict]:
    """Return real interfaces available for capture (no placeholders)."""
    if IS_WINDOWS:
        if not DUMPCAP.exists():
            raise CaptureError(
                f"dumpcap.exe not found at {DUMPCAP}. Install/repair Wireshark."
            )
        proc = subprocess.run(
            [str(DUMPCAP), "-D"], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            raise CaptureError(f"dumpcap -D failed: {proc.stderr.strip()}")
        interfaces = []
        for line in proc.stdout.splitlines():
            m = re.match(r"\s*(\d+)\.\s+(\S+)\s*(?:\((.+)\))?", line)
            if m:
                idx, device, friendly = m.groups()
                interfaces.append({
                    "index": int(idx),
                    "device": device,
                    "name": friendly or device,
                })
        return interfaces
    else:
        proc = subprocess.run(
            ["tcpdump", "-D"], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            raise CaptureError(f"tcpdump -D failed: {proc.stderr.strip()}")
        interfaces = []
        for line in proc.stdout.splitlines():
            m = re.match(r"\s*(\d+)\.(\S+)\s*(?:\[(.*)\])?", line)
            if m:
                idx, device, desc = m.groups()
                interfaces.append({
                    "index": int(idx),
                    "device": device,
                    "name": desc or device,
                })
        return interfaces


@dataclass
class CaptureSession:
    session_id: str
    interface: str
    pcap_path: Path
    process: subprocess.Popen
    start_time: float
    end_time: Optional[float] = None
    packet_count: Optional[int] = None
    error: Optional[str] = None
    packet_target: Optional[int] = None
    duration_target: Optional[int] = None
    # "duration_target_reached" | "packet_target_reached" | "target_reached"
    # | "user_stopped" | "safety_timeout" | None (still running)
    stop_reason: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.end_time is None:
            return "CAPTURING"
        return "STOPPED"

    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    def to_dict(self) -> dict:
        # While still capturing, packet_count is refreshed from the real
        # (still-being-written) file each call — same "no I/O caching,
        # always read the real current state" pattern read_packets() uses
        # for the packet table. Once stopped, self.packet_count (set from
        # capinfos in _finalize()) is authoritative and this is skipped.
        live_count = self.packet_count
        if self.status() == "CAPTURING":
            live_count = peek_live_packet_count(self.pcap_path)
        return {
            "session_id": self.session_id,
            "interface": self.interface,
            "pcap_path": str(self.pcap_path),
            "status": self.status(),
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds(),
            "packet_count": live_count,
            "packet_target": self.packet_target,
            "duration_target": self.duration_target,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


def _capfile_stats(pcap_path: Path) -> dict:
    """Real packet count/duration read back from the capture file itself."""
    if IS_WINDOWS and CAPINFOS.exists():
        proc = subprocess.run(
            [str(CAPINFOS), "-c", "-u", "-M", str(pcap_path)],
            capture_output=True, text=True, timeout=30,
        )
        out = proc.stdout
        stats = {"packet_count": None, "duration_seconds": None}
        m = re.search(r"Number of packets:\s*([\d,]+)", out)
        if m:
            stats["packet_count"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Capture duration:\s*([\d.]+) seconds", out)
        if m:
            stats["duration_seconds"] = float(m.group(1))
        return stats
    proc = subprocess.run(
        ["capinfos", "-c", "-u", "-M", str(pcap_path)],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode == 0:
        out = proc.stdout
        stats = {"packet_count": None, "duration_seconds": None}
        m = re.search(r"Number of packets:\s*([\d,]+)", out)
        if m:
            stats["packet_count"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Capture duration:\s*([\d.]+) seconds", out)
        if m:
            stats["duration_seconds"] = float(m.group(1))
        return stats
    return {"packet_count": None, "duration_seconds": None}


def validate_and_stat_pcap(pcap_path: Path) -> dict:
    """
    Strict PCAP validation for uploaded files: unlike `_capfile_stats`
    (used by the live-capture stop path, which tolerates capinfos being
    briefly unavailable), this raises CaptureError if the file genuinely
    isn't a readable capture — used to reject malformed PCAP uploads
    before wasting time on extraction. Real packet count only; never
    fabricated.
    """
    pcap_path = Path(pcap_path)
    capinfos = str(CAPINFOS) if (IS_WINDOWS and CAPINFOS.exists()) else "capinfos"
    try:
        proc = subprocess.run(
            [capinfos, "-c", "-u", "-M", str(pcap_path)],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        raise CaptureError("capinfos not found; cannot validate uploaded PCAP.")
    except subprocess.TimeoutExpired:
        raise CaptureError("Timed out validating uploaded PCAP.")

    # capinfos exits non-zero and warns "appears to have been cut short in
    # the middle of a packet... will continue anyway" for any PCAP whose
    # last write wasn't flushed cleanly — which in practice is every file
    # this app's own live-capture stop_capture() produces (Popen.terminate()
    # doesn't give dumpcap a chance to close the file gracefully on
    # Windows). capinfos still parses and reports real stats despite the
    # warning, and cicflowmeter/scapy reads these files fine, so treat a
    # recovered packet count as valid. Only reject when capinfos truly
    # couldn't read anything (garbage bytes, wrong format, empty file).
    stats = {"packet_count": None, "duration_seconds": None}
    m = re.search(r"Number of packets:\s*([\d,]+)", proc.stdout)
    if m:
        stats["packet_count"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Capture duration:\s*([\d.]+) seconds", proc.stdout)
    if m:
        stats["duration_seconds"] = float(m.group(1))

    if stats["packet_count"] is None:
        raise CaptureError(
            "File is not a valid PCAP/PCAPNG capture: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    if stats["packet_count"] == 0:
        raise CaptureError("PCAP contains no packets.")
    return stats


MAX_PACKET_READ = 50_000  # hard ceiling for a single read_packets() call


def read_packets(pcap_path: Path, limit: int = 100, offset: int = 0) -> list[dict]:
    """
    Real per-packet rows read back from the actual PCAP file via tshark
    (No., Time, Source, Destination, Protocol, Src/Dst Port, Length, Info).
    Returns [] rather than fake rows if the file can't be read yet (e.g.
    mid-write) or tshark isn't available.

    Supports paging into large captures (tens of thousands of packets)
    without ever pulling the whole file into memory at once: `offset`/
    `limit` are pushed down to tshark itself as a frame-number RANGE
    display filter (`-Y "frame.number > offset && frame.number <=
    offset+limit"`), so only the requested page's worth of fields is ever
    parsed into Python objects — the caller is expected to page through a
    capture rather than request it all in one call (limit is hard-capped
    at MAX_PACKET_READ for that reason).

    Deliberately does NOT combine `-c` with `-Y`: in this tshark build
    (3.0.1) `-c` counts packets *read* from the file, not packets that
    pass the display filter, so `-Y "frame.number > 100" -c 5` silently
    returns nothing (it stops after reading the first 5 packets, none of
    which are past frame 100) instead of erroring — verified directly
    against a real capture. Bounding both ends of the range purely via
    the filter avoids that entirely.
    """
    limit = max(1, min(limit, MAX_PACKET_READ))
    tshark = TSHARK if IS_WINDOWS else Path("tshark")
    if IS_WINDOWS and not TSHARK.exists():
        return []

    cmd = [str(tshark), "-r", str(pcap_path), "-T", "fields"]
    for f in _PACKET_FIELDS:
        cmd += ["-e", f]
    cmd += ["-E", "header=n", "-E", "separator=\t", "-E", "quote=n"]
    upper = offset + limit
    frame_filter = f"frame.number <= {upper}" if offset <= 0 else f"frame.number > {offset} && frame.number <= {upper}"
    cmd += ["-Y", frame_filter]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=90,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    # A non-zero exit here is routinely just tshark's "appears to have
    # been cut short in the middle of a packet" warning (this app's own
    # captures always end this way — see stop_capture()'s docstring in
    # this module) — it still finishes emitting every packet it could
    # read before the truncated tail. Only truly empty stdout means no
    # usable data came back.
    if not proc.stdout.strip():
        return []

    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != len(_PACKET_FIELDS):
            continue
        (no, t, ip_src, ip6_src, ip_dst, ip6_dst, proto,
         tcp_sp, udp_sp, tcp_dp, udp_dp, length, info) = parts
        rows.append({
            "no": no,
            "time": round(float(t), 6) if t else None,
            "source": ip_src or ip6_src or "",
            "destination": ip_dst or ip6_dst or "",
            "protocol": proto,
            "src_port": tcp_sp or udp_sp or "",
            "dst_port": tcp_dp or udp_dp or "",
            "length": int(length) if length else None,
            "info": info,
        })
    return rows


def start_capture(
    interface_device: str,
    pcaps_dir: Path,
    duration_seconds: Optional[int] = DEFAULT_CAPTURE_DURATION_SECONDS,
    packet_target: Optional[int] = DEFAULT_CAPTURE_PACKET_TARGET,
) -> CaptureSession:
    """
    packet_target (default 10000) is passed straight to dumpcap/tcpdump's
    own `-c` autostop flag, so the capture tool stops itself once that
    many packets have been written — this is the default target callers
    care about ("capture up to 10000 packets").

    duration_seconds (default 300) is passed to dumpcap's own
    `-a duration:N` autostop condition on Windows (tcpdump has no
    equivalent native flag, so on Linux, and as a cross-platform backstop
    on every platform, check_and_finalize() enforces duration_target by
    polling session.duration_seconds() and stopping the process itself).
    It exists as an upper bound so a capture can't run indefinitely on a
    quiet interface that never reaches packet_target, and wide enough
    (300s) to give a normal interface realistic time to actually reach
    10000 packets before duration cuts it off first.
    """
    pcaps_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    session_id = uuid.uuid4().hex[:8]
    pcap_path = pcaps_dir / f"capture_{timestamp}_{session_id}.pcap"

    if IS_WINDOWS:
        if not DUMPCAP.exists():
            raise CaptureError(f"dumpcap.exe not found at {DUMPCAP}")
        cmd = [str(DUMPCAP), "-i", interface_device, "-w", str(pcap_path)]
        if duration_seconds:
            cmd += ["-a", f"duration:{duration_seconds}"]
    else:
        # No `sudo` here on purpose: a backend service has no TTY for a
        # sudo password prompt and would hang. Run the backend as root,
        # or (recommended) grant tcpdump capture capabilities once via:
        #   sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump)
        # See README.md "Linux setup".
        cmd = ["tcpdump", "-i", interface_device, "-w", str(pcap_path)]
        # No native duration flag on tcpdump — check_and_finalize() below
        # enforces duration_target via polling instead.

    if packet_target:
        cmd += ["-c", str(packet_target)]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as e:
        raise CaptureError(f"Capture tool not found: {e}")

    time.sleep(0.5)
    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise CaptureError(
            f"Capture process exited immediately (code {proc.returncode}): "
            f"{stderr.strip()}"
        )

    return CaptureSession(
        session_id=session_id,
        interface=interface_device,
        pcap_path=pcap_path,
        process=proc,
        start_time=time.time(),
        packet_target=packet_target,
        duration_target=duration_seconds,
    )


def peek_live_packet_count(pcap_path: Path) -> Optional[int]:
    """Best-effort real packet count read back from a capture file that is
    still being actively written by dumpcap/tcpdump — same tolerant
    handling as _capfile_stats (a mid-write file routinely makes capinfos
    exit non-zero; still parse whatever count it did report rather than
    guessing or returning a stale number)."""
    if not pcap_path.exists() or pcap_path.stat().st_size == 0:
        return None
    return _capfile_stats(pcap_path)["packet_count"]


def _finalize(session: CaptureSession) -> CaptureSession:
    """Shared tail end of both a user-initiated stop and an automatic one
    (capture tool exited on its own after reaching packet_target, or the
    safety timeout fired) — reads back the real, final packet count via
    capinfos exactly once the process has actually exited."""
    session.end_time = time.time()

    if not session.pcap_path.exists() or session.pcap_path.stat().st_size == 0:
        session.error = "Capture stopped but no PCAP data was written."
        return session

    stats = _capfile_stats(session.pcap_path)
    session.packet_count = stats["packet_count"]
    return session


def stop_capture(session: CaptureSession) -> CaptureSession:
    """User-initiated stop. Safe to call even if the capture process
    already exited on its own (e.g. it hit packet_target) — in that case
    this just finalizes without needing to terminate anything, and
    stop_reason is left as whatever check_and_finalize() already set
    rather than being overwritten to "user_stopped"."""
    if session.status() != "CAPTURING":
        return session

    if session.process.poll() is None:
        # Only stamp "user_stopped" if nothing has already given this
        # stop a more specific reason (check_and_finalize() sets
        # duration_target_reached/safety_timeout on the session *before*
        # calling us to actually terminate the process).
        if session.stop_reason is None:
            session.stop_reason = "user_stopped"
        try:
            session.process.terminate()
            session.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            session.process.kill()
            session.process.wait(timeout=5)
    elif session.stop_reason is None:
        session.stop_reason = "target_reached"

    return _finalize(session)


def check_and_finalize(session: CaptureSession) -> CaptureSession:
    """Call on every status poll while a session is CAPTURING. Detects
    things that can happen between polls, since nothing else watches the
    subprocess in the background:
      1. The capture tool exited on its own. On Windows this means it hit
         either `-a duration:N` (duration_target) or `-c packet_target` —
         both are dumpcap's own native stop conditions, so we distinguish
         which one fired by comparing the elapsed time and live packet
         count against each target rather than counting packets
         ourselves.
      2. duration_target elapsed but the process is still running — only
         reachable on a platform without a native duration stop (e.g.
         tcpdump on Linux, which has no equivalent of `-a duration:N`);
         on Windows dumpcap already self-exits via case 1 first. Stopped
         here rather than left to run past its real target.
      3. The safety timeout elapsed — a backstop for a capture that
         somehow outlives both of its own targets (e.g. no targets set
         at all); genuinely terminated here, not just marked as stopped.
    Returns the session unchanged if none of these apply yet."""
    if session.status() != "CAPTURING":
        return session

    if session.process.poll() is not None:
        elapsed = session.duration_seconds()
        if session.duration_target and elapsed >= session.duration_target:
            session.stop_reason = "duration_target_reached"
        elif session.packet_target is not None:
            session.stop_reason = "packet_target_reached"
        else:
            session.stop_reason = "target_reached"
        return _finalize(session)

    if session.duration_target and session.duration_seconds() >= session.duration_target:
        session.stop_reason = "duration_target_reached"
        return stop_capture(session)

    if session.duration_seconds() >= CAPTURE_SAFETY_TIMEOUT_SECONDS:
        session.stop_reason = "safety_timeout"
        return stop_capture(session)

    return session
