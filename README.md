# NIDS: Network State Detection and Forecasting

A network intrusion detection and next-window forecasting pipeline with a
Wireshark-inspired desktop UI. Its primary path is:

```
PCAP Capture → CICFlowMeter → Current State → LSTM Forecast → ATT&CK Context
```

The original ANN classifies each flow. A separate LSTM forecasts the next
aggregate traffic state and an offline, evidence-aware rules layer supplies
restrained MITRE ATT&CK candidates. The project does not claim attribution,
intent, or host-level confirmation. See `PIPELINE.md` for the pre-build audit.

## Architecture

```
Live Network Interface
        │  dumpcap (Windows) / tcpdump (Linux)
        ▼
   pcaps/*.pcap
        │  cicflowmeter (Python, Scapy-based)
        ▼
  features/*.csv  (82 columns: 5 ID/label + 77 model features)
        │  MinMaxScaler (models/minmax.bin) → ANN (models/ISAA_ANN.h5)
        ▼
 results/*_prediction.json   (per-flow BENIGN/DDoS/DoS/PortScan + confidence)
```

Backend: FastAPI (`backend/api/main.py`) exposes this as a state machine —
`IDLE → CAPTURING → CAPTURE_COMPLETED → EXTRACTING → EXTRACTION_COMPLETED
→ PREDICTING → PREDICTION_COMPLETED`, or `ERROR` on any failure — and
serves the static frontend. Frontend: plain HTML/CSS/JS
(`frontend/`), no build step, polls the backend every ~1.2s.

## Offline upload mode

A second, independent input mode alongside live capture: upload a
`.pcap`/`.pcapng`/`.csv` file for the same extraction/prediction pipeline
to run against, without touching any network interface. Toggle it via the
**LIVE CAPTURE / UPLOAD FILE** switch in the toolbar. Each upload gets an
isolated session directory (`uploads/sessions/<session_id>/{input,features,results}`)
with a server-generated filename — the client's filename is never used to
build a filesystem path, only kept as a display string.

State machine (separate from, and independent of, the live-capture one):
`FILE_UPLOADED → VALIDATING → EXTRACTING (PCAP only) → PREDICTING →
PREDICTION_COMPLETED`, or `ERROR`. A CSV upload skips `EXTRACTING` and
goes straight from `VALIDATING` to `PREDICTING` — CICFlowMeter never runs
on an already-extracted CSV.

API: `POST /api/upload` (multipart), `GET /api/upload/{id}/status`,
`GET /api/upload/{id}/download` (the original upload plus a
`Current_State` column), `GET /api/upload/sessions`. Reuses
`backend/capture/capture.py` (PCAP validation via `capinfos`),
`backend/extraction/extract.py`, and `backend/prediction/predict.py`
unchanged — no second CICFlowMeter or feature-matching implementation.

## Temporal dataset pipeline

`backend/temporal/` turns a `Current_State`-labelled feature CSV into a
windowed, sequenced dataset for temporal modeling. It does not change
the existing ANN. Pipeline: parse/validate timestamps → sort
chronologically → fixed time windows (default 10s) → aggregate each
window's flows into a 28-feature network-state vector (flow/byte/packet
counts, TCP flags, IAT stats, unique endpoints — see
`backend/temporal/schema.py`, reusing `features.py`'s column matcher, not
duplicating it) → dominant state label + `attack_present` flag + per-class
counts → `state_transitions.csv` (S_t → S_t+1) → sliding sequences
(default length 5) → chronological (never random) 70/15/15 train/val/test
split → a separate `models/temporal_scaler.bin` fit only on the training
split. Outputs land in `data/temporal/<session>/`.

CLI: `python -m backend.temporal.temporal_dataset --input features/<csv> --output data/temporal/<name> [--window-size 10] [--sequence-length 5]`.
API (secondary, doesn't touch the capture/extract/predict routes):
`POST /api/temporal/prepare {csv_path}`, `GET /api/temporal/status`.
Tests: `pytest tests/test_temporal.py`.

## LSTM next-window forecasting

`backend/lstm/` adds an independent four-state forecaster over the ANN's
outputs. It analyzes the four configured CICIDS2017 sessions in chunks,
scores every valid row through the existing ANN/scaler/feature matcher,
aggregates deterministic 10-row proxy windows into the existing 28-feature
schema, and builds `5 × 28` sequences targeting the next window. Source files
are read-only and remain external. Set `NIDS_CICIDS2017_DIR` to the directory
containing the required Monday, Tuesday, Wednesday, and Thursday-Afternoon-
Infiltration CSVs.

Because those four exports do not provide a usable timestamp for this task,
one row is explicitly treated as one proxy second. Sessions, splits, and gaps
remain hard boundaries. `INVALID_FEATURES` rows never vote for a target;
windows with no scoreable states are omitted. Missing source/destination IP
and source-port cardinalities are recorded as unavailable zeros in metadata.

Training uses `LSTM(64) → Dropout(0.3) → Dense(32, ReLU) → Softmax(4)`, seed
42, three expanding chronological folds, an untouched terminal 15% holdout,
and compares unweighted versus capped class-weighted fitting by mean rolling
macro-F1. Persistence and multinomial logistic regression use identical
sequences/splits. Artifacts and reports are written under
`artifacts/lstm_forecaster/<version>/`; compact fingerprinted windows are
cached under `data/lstm_cache/`.

The published artifact inventory and interpretation guidance are documented in
`artifacts/lstm_forecaster/README.md`. `v1-3a5264a499ed` is the active model;
older versions are retained only as historical evidence of the scaler-leakage
defect and are unsuitable for normal metric interpretation.

The four CICIDS2017 source CSVs used by Phase 3 are published separately as a
GitHub Release asset because the raw files exceed GitHub's normal Git-object
limit. Download, checksum, and setup instructions are in `datasets/README.md`.

API: `POST /api/lstm/train {"force_rebuild": false}`,
`GET /api/lstm/status`, `POST /api/lstm/forecast`, and
`GET /api/lstm/report`. The Temporal Dataset → **LSTM FORECAST** tab shows
bounded totals, progress, holdout/baseline metrics, a one-window forecast,
probabilities, evaluation status, and the row-order proxy warning. Tests:
`pytest tests/test_lstm_forecasting.py`.

## Phase 4 direct H1–H6 forecasting

Phase 4 is separate from the active Phase 3 one-step model. It consumes the
same observed `5 × 28` histories but directly predicts six four-class targets
with `LSTM(64) → Dropout(0.3) → Dense(32, ReLU) → Dense(24) → Reshape(6,4)`.
No predicted class vector is recursively substituted for a future 28-feature
state. Artifacts live under `artifacts/lstm_multistep/<version>/`; Phase 3 files
under `artifacts/lstm_forecaster/` are unchanged.

The locked whole-session split follows official CICIDS2017 chronology: Monday
through Thursday afternoon for training, Friday morning plus Friday PortScan
for validation, and Friday DDoS as the untouched test. The validation-selected
warning threshold is frozen before test scoring. Since the CSVs have no usable
timestamps here, outputs are labelled `+1 window` through `+6 windows` and the
API returns `seconds_ahead: null`.

Commands:

```bash
python scripts/multistep.py dataset
python scripts/multistep.py train
python scripts/multistep.py evaluate
python scripts/multistep.py benchmark
```

API: `POST /api/lstm/forecast/multistep`. The **MULTI-STEP FORECAST** tab shows
H1–H6 state probabilities, the frozen early-warning decision, separate MITRE
mapping confidence, and an inline trajectory chart. Reports are under
`reports/multistep_*`; tests are in `tests/test_lstm_multistep.py` and
`tests/test_multistep_api.py`.

Measured limitation: the strict training and validation sessions contain no
ANN-derived DDoS targets, while Friday DDoS is dominated by them. This is the
intended untouched generalization test, not a representative four-class fit;
unsupported validation metrics are `N/A`, test classes with fewer than 30
samples are flagged low-support, and the reports must be read alongside the
binary attack metrics and onset misses.

## Temporal dataset validation

`backend/temporal/validate.py` re-inspects an already-prepared temporal
dataset (read-only — never modifies the source CSV/PCAP or any temporal
artifact) and produces a structured report across 10 checks: Current
State, Timestamps, 10s Windows, 28 Features, Transitions, Sequences,
Chronological Split, Data Leakage, Missing Data, Duplicates — each
`PASS`/`WARNING`/`FAIL`/`NOT_AVAILABLE`. Leakage checking covers exact
cross-split sequence duplicates *and* overlapping-window sequences (hash/
set-based, O(n) — never pairwise `O(n²)`), plus a structural scaler- and
label-leakage check. Designed for large CSVs: the flow-level CSV is read
once and reused across every raw-level check; a synthetic 50,000-row
fixture is covered in the test suite (prepare+validate complete in
seconds, not minutes).

API: `POST /api/temporal/validate` (validates whatever the server's last
completed `/api/temporal/prepare` produced — no request body needed),
`GET /api/temporal/validate/status`, `GET /api/temporal/validate/report`
(downloads the persisted `validation_report.json`). UI: Temporal Dataset
→ **VALIDATION** tab, plus a `[TEMPORAL STATES] [TRANSITIONS] [SEQUENCES]
[VALIDATION]` sub-tab set in the main table area. `/api/pipeline/reset`
also clears temporal/validation state (no files deleted).
Tests: `pytest tests/test_temporal_validate.py`.

## Repository layout

```
backend/
  api/main.py            FastAPI app, state machine, all /api/* routes
  capture/capture.py      dumpcap/tcpdump wrapper + real packet-table reader (tshark)
  extraction/extract.py   cicflowmeter (pcap -> csv) wrapper
  prediction/predict.py   scaler + ANN inference (the original bug, fixed) + Current_State CSV write-back
  prediction/features.py  77-feature schema + name-normalizing column matcher
  lstm/                    chunked proxy-window prep, rolling evaluation, Keras forecaster, job state
  upload/manager.py       offline PCAP/CSV upload sessions + state machine
  requirements.txt
config/config.json
frontend/                index.html, styles.css, app.js
models/                  ISAA_ANN.h5, minmax.bin (unchanged, from the original repo)
notebooks/               original training/inference notebooks (reference only)
pcaps/  features/  results/  uploads/sessions/
scripts/
  packetsniff.sh          original tcpdump script (kept for manual/reference use)
  patch_cicflowmeter.py   fixes 2 upstream bugs in the cicflowmeter package
  run_backend.ps1 / .sh   one-command setup + start
PIPELINE.md               audit of the pre-existing repo (read this first)
```

## Why `cicflowmeter` (Python) instead of the Java CICFlowMeter

The original Java CICFlowMeter (`ahlashkari/CICFlowMeter`) depends on
`jnetpcap`, a native library with no working 64-bit Windows/modern-JDK
build path and no CLI without WinPcap dev headers — a dead end on this
target platform. In its place, the MIT-licensed `cicflowmeter` PyPI
package (Scapy-based) computes the same flow-feature set with no native
dependency. It ships with two bugs that break every invocation
(`create_sniffer()` positional-arg mismatch; a hard dependency on a real
`tcpdump` binary just to compile a redundant BPF filter) —
`scripts/patch_cicflowmeter.py` fixes both, idempotently, against
whatever environment it's pointed at. `run_backend.ps1`/`.sh` run it
automatically.

CICFlowMeter's own output column names have also changed across versions
(`Total Fwd Packets` vs `Tot Fwd Pkts` vs this build's `tot_fwd_pkts`, plus
one outright naming inconsistency: the original CICIDS2017 dataset calls
a column "CWE Flag Count" for what is actually the TCP **CWR** flag).
`backend/prediction/features.py` aligns columns by a normalized token
signature (abbreviations expanded, order-independent) rather than trusting
literal names or column position — the bug in the original prediction
notebook.

## Installation

**Requires:** Python 3.11 or 3.12 (TensorFlow 2.18 does not yet support
3.13+ — this project's venv was built with 3.12), Windows, Linux, or
macOS. Java is **not** required (the Java CICFlowMeter path was abandoned
— see above).

### Windows

1. Install [Wireshark](https://www.wireshark.org/) (includes Npcap and
   `dumpcap.exe`/`tshark.exe`/`capinfos.exe`, which this project calls
   directly — default install path `C:\Program Files\Wireshark`).
   Non-admin capture works out of the box unless Npcap was installed with
   "restrict to Administrators" checked.
2. From the repo root: `scripts\run_backend.ps1` — creates the venv,
   installs `backend/requirements.txt`, applies the cicflowmeter patch,
   and starts the server.

### Linux

1. Install `tcpdump` and Wireshark's CLI tools (`tshark`, `capinfos` — the
   `wireshark-common`/`tshark` package on most distros).
2. Grant non-root capture once (avoids running the backend as root, and
   avoids `sudo` hanging on a password prompt inside a service):
   ```
   sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump)
   ```
3. From the repo root: `bash scripts/run_backend.sh`

### macOS

macOS has `tcpdump` built in but `/dev/bpf*` is root-only and there is no
`setcap`, so an unprivileged backend cannot capture until Wireshark's
**ChmodBPF** daemon makes `/dev/bpf*` group-`admin` readable at boot.

1. One-time setup (installs the Wireshark cask — which also provides
   `tshark`/`capinfos`, needed for the packet table, packet-count/duration,
   and `.pcap` upload — and loads ChmodBPF):
   ```
   bash scripts/macos_setup.sh
   ```
   Then **log out and back in** (or reboot) so the `admin` group grant
   takes effect. Verify with `ls -l /dev/bpf0` (group should be `admin`,
   with a read bit) and `tcpdump -c1 -i en0` (should succeed with no
   `sudo`).
2. From the repo root: `bash scripts/run_backend.sh` (the same script the
   Linux path uses — `python3 -m venv` + `uvicorn`).

Without step 1, live capture fails with
`(cannot open BPF device) /dev/bpf0: Permission denied` and `.pcap`
uploads fail on a missing `capinfos`. As a last resort you can skip
Wireshark and either run `sudo chmod +r /dev/bpf*` after every boot or
start the backend under `sudo`.

The toolbar interface dropdown shows friendly names on macOS
(`en0 — Wi-Fi`, `en7 — USB 10/100/1000 LAN`, `lo0 — Loopback`, …), mapped
from `networksetup -listallhardwareports`; devices macOS has no name for
(`utun*`, `awdl0`, `llw0`, …) keep the raw `tcpdump -D` flag text.

### Common (either OS)

Backend deps: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`,
`joblib`, `tensorflow==2.18.0`, `cicflowmeter==0.5.0`, `scapy==2.6.1` (all
in `backend/requirements.txt`). Frontend: none — static HTML/CSS/JS,
served by the backend.

## Running

```
scripts\run_backend.ps1      # Windows
bash scripts/run_backend.sh  # Linux and macOS
```

Then open **http://127.0.0.1:8765** — the backend serves the UI directly.

## Performing a real capture

1. Pick an interface from the toolbar dropdown (populated from a real
   `dumpcap -D` / `tcpdump -D` call — not a fixed list).
2. **Start Capture** → status turns `CAPTURING`, duration/packet count
   update live, packet table fills from real per-packet `tshark` reads of
   the growing PCAP.
3. **Stop Capture** → the process is terminated, the PCAP is finalized,
   and packet count/duration are read back from the file itself via
   `capinfos` (not estimated).
4. **Extract Features** → runs `cicflowmeter` in-process against the PCAP;
   the CSV path, flow count, feature count, and elapsed time shown are all
   read from the real output file.
5. **Run Prediction** → loads `models/ISAA_ANN.h5` + `models/minmax.bin`,
   scales and classifies every flow, shows the real per-class breakdown
   and overall traffic state. Also written to
   `results/<capture>_prediction.json`.

At every stage, a failure (bad interface, empty PCAP, no flows extracted,
missing model, unmatched feature columns, etc.) surfaces the real
exception message in the pipeline strip and the relevant detail card —
never a fabricated value. Where a value truly isn't available yet, the UI
shows `N/A`.

## Test result (verified during this build)

Full pipeline run on this machine, real Wi-Fi traffic, no synthetic data:

| Stage | Result |
|---|---|
| Capture | 907 packets in 4.05s (also verified: 219 pkts/3.36s, 514 pkts/4.23s, 354 pkts/4.17s across separate runs) |
| Extraction | 82 columns, 14–21 flows per run, 0.22s–1.76s |
| Prediction | Real per-flow softmax output, non-constant (e.g. 0.4638–0.9958 confidence across flows in one run) — confirms the scaler-application fix actually took effect |

Also verified via the HTTP API directly (not just the underlying Python
modules) and via a full browser pass (Playwright): interface dropdown
populates with real adapter names, Start/Stop/Extract/Predict button
states track the real pipeline stage, no browser console errors. Error
paths verified: invalid interface (real `dumpcap` error surfaced, HTTP
500), predicting with nothing extracted yet (HTTP 400), extracting a
nonexistent PCAP (real "PCAP not found" error, pipeline → `ERROR`).

## Known limitations

- **Model accuracy on real-world traffic**: `ISAA_ANN` was trained solely
  on CICIDS2017 (2017 lab-generated attack traffic). In testing, it
  frequently classified ordinary modern HTTPS/QUIC/STUN traffic as
  DDoS/DoS — a generalization limitation of the existing model, not a
  pipeline bug. Per the task scope, no new model was trained to address
  this.
- **Random Forest is not available for inference**: the original repo's
  `notebooks/NIDS - ML.ipynb` trains and scores Random Forest/Decision
  Tree/KNN but never persists any of them (no `.pkl`/`.joblib` saved) —
  only the ANN was ever saved to disk, so only the ANN is wired into the
  API.
- **`sklearn` version skew**: `models/minmax.bin` was pickled with an
  older scikit-learn (1.0.2); it still loads and functions correctly
  under 1.5.2 but prints an `InconsistentVersionWarning`.
- **Packet table is a periodic snapshot**, not a true packet-by-packet
  live stream: it re-reads the PCAP via `tshark` on each poll
  (~1.5s interval) rather than tailing packets as they arrive.
- **Windows-only auto packet-count/duration** via `capinfos.exe`; the
  Linux and macOS paths call the same tool name from `PATH` and need
  `tshark`/`capinfos` installed (Linux: `wireshark-common`/`tshark`;
  macOS: the Wireshark cask — see `scripts/macos_setup.sh`). Without them
  the packet table stays empty and packet count/duration show `N/A`, and
  the FILE ANALYSIS `.pcap` path fails its `capinfos` validation.
