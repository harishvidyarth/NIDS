#!/usr/bin/env python3
"""Populate every XDR prototype panel from repository-local sample data."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8765"


def request_json(base: str, path: str, payload=None, token: str | None = None, method: str | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-NIDS-Response-Token"] = token
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method or ('POST' if data else 'GET')} {path}: HTTP {error.code} {detail}") from error


def upload_csv(base: str, path: Path) -> dict:
    boundary = "----nids-xdr-" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/api/upload", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


def wait_for(base: str, path: str, terminal: set[str], timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = request_json(base, path)
        if str(result.get("stage", "")).upper() in terminal:
            return result
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {path}")


def sample_duration(path: Path) -> float:
    timestamps = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = next((item for key, item in row.items() if key.strip().lower() == "timestamp"), None)
            if value:
                try:
                    timestamps.append(dt.datetime.fromisoformat(value.strip()))
                except ValueError:
                    continue
    return (max(timestamps) - min(timestamps)).total_seconds() if timestamps else 0.0


def start_server(base: str) -> None:
    if base != DEFAULT_URL:
        raise RuntimeError("--serve currently uses the documented 127.0.0.1:8765 address.")
    env = dict(os.environ)
    env["NIDS_XDR_DEMO"] = "1"
    command = [str(ROOT / ".venv-lstm312" / "bin" / "python"), "-m", "uvicorn",
               "backend.api.main:app", "--host", "127.0.0.1", "--port", "8765"]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            request_json(base, "/api/pipeline")
            print(f"Started backend PID {process.pid}; it remains available for dashboard inspection.")
            return
        except Exception:
            if process.poll() is not None:
                raise RuntimeError("Backend exited during startup.")
            time.sleep(0.5)
    raise RuntimeError("Backend did not become ready.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="start uvicorn with the repository Python 3.12 environment")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    if args.serve:
        start_server(args.url)

    capabilities = request_json(args.url, "/api/response/capabilities")
    token = capabilities["local_authorization_token"]
    request_json(args.url, "/api/xdr/demo/enable", {}, token, "POST")

    samples = sorted((ROOT / "features").glob("*.csv"), key=sample_duration, reverse=True)
    if not samples:
        raise RuntimeError("No sample feature CSV exists under features/.")
    upload = upload_csv(args.url, samples[0])
    session_id = upload["session_id"]
    upload_status = wait_for(args.url, f"/api/upload/{session_id}/status", {"PREDICTION_COMPLETED", "ERROR"})
    if upload_status["stage"] == "ERROR":
        raise RuntimeError(upload_status.get("error") or "Sample upload analysis failed.")

    logs = [json.loads(line) for line in (ROOT / "backend/ingest/sample_zeek/sample.log").read_text().splitlines() if line.strip()]
    # One explicit later-capture edge makes baseline comparison visible even
    # when the operator has already learned the bundled 200-row sample.
    logs.append({
        "_path": "conn", "ts": 1700040000, "uid": "DEMO-LATERAL",
        "id.orig_h": "10.0.0.20", "id.orig_p": 49152,
        "id.resp_h": "10.99.0.77", "id.resp_p": 445,
        "proto": "tcp", "service": "smb", "orig_bytes": 1200, "resp_bytes": 80,
    })
    ingest = request_json(args.url, "/api/ingest/zeek", {"session_id": session_id, "logs": logs}, token, "POST")
    request_json(args.url, "/api/deception/canary")

    temporal_error = None
    try:
        request_json(args.url, "/api/temporal/prepare", {"csv_path": upload_status["processed_csv_path"]}, method="POST")
        temporal = wait_for(args.url, "/api/temporal/status", {"COMPLETED", "ERROR"})
        if temporal["stage"] == "ERROR":
            temporal_error = temporal.get("error")
    except RuntimeError as error:
        temporal_error = str(error)

    forecast = {}
    forecast_error = None
    try:
        forecast = request_json(args.url, "/api/lstm/forecast/multistep", {}, method="POST")
    except RuntimeError as error:
        forecast_error = str(error)

    graph = request_json(args.url, f"/api/graph?session_id={session_id}", token=token)
    triage = request_json(args.url, f"/api/triage?session_id={session_id}", {}, token, "POST")
    plan = request_json(args.url, "/api/response/plan", {
        "target": "198.51.100.77", "ttl_seconds": 900,
        "verdict": {"attack_class": "DoS", "signature_verdict": "DoS"},
        "forecast": forecast,
    }, token, "POST")
    hits = request_json(args.url, "/api/deception/hits", token=token)
    audit = request_json(args.url, "/api/response/audit", token=token)

    rows = [
        ("Uploaded sample", f"{samples[0].name} ({upload_status.get('flow_count', 0)} flows)"),
        ("Zeek records", ingest["record_count"]),
        ("DNS entropy", f"{ingest['enrichment']['dns_query_entropy_mean']:.3f}"),
        ("Beacon score", f"{ingest['enrichment']['beacon_score_max']:.3f}"),
        ("JA3 novelty", f"{ingest['enrichment']['ja3_novelty']:.3f}"),
        ("Graph", f"{len(graph['nodes'])} nodes, {len(graph['surprising_edges'])} surprising edges"),
        ("Campaign score", f"{graph['campaign_score']:.3f}"),
        ("Triage", f"{triage['confidence']} · {triage['source']}"),
        ("Forecast timing", f"{forecast.get('time_to_attack_seconds')} seconds" if forecast else f"unavailable ({forecast_error})"),
        ("Temporal", "COMPLETED" if not temporal_error else f"unavailable ({temporal_error})"),
        ("Response", f"{plan['step']} · DRY RUN · {plan['command']}"),
        ("Deception hits", len(hits["hits"])),
        ("Audit events", len(audit["events"])),
    ]
    width = max(len(label) for label, _ in rows)
    print("\nXDR prototype demo")
    print("=" * 72)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    print("=" * 72)
    print(f"Dashboard: {args.url}  →  XDR / CAMPAIGN")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
