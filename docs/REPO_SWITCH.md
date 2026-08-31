# SIH26153 — Repo switch: network-attack-forecasting → NIDS

> Decision 2026-08-31. The SIH 2026 PS 26153 submission now builds on **NIDS**
> (`github.com/harishvidyarth/NIDS`), not `network-attack-forecasting`
> (`github.com/harishvidyarthcsecs/network-attack-forecasting`, being retired).

## What moved here

- `docs/PROBLEM_STATEMENT.md`, `docs/PRIOR_ART.md`,
  `docs/SIH_COMPLETION_ANALYSIS.md`, `docs/SIH_TEAM_PLAN.md`
- `scripts/build_sih_tracker.py` + `scripts/requirements-tracker.txt` +
  `scripts/sih_tracker_data/` (Google Sheets tracker, live sheet
  `1228SXz-TP6SUP9hXsaHH9GvdE9RHyEVcxT2-yTI5-0A`)

## Re-score pending

`docs/SIH_COMPLETION_ANALYSIS.md` (the 69% and deviations D1–D6) was scored
against `network-attack-forecasting`. It must be re-scored against NIDS. NIDS's
own known gaps to fold into the deviation register:

- Phase-3 LSTM terminal test holdout is **BENIGN-only** (29,339 samples, 0
  attacks) — macro-F1 0.25, attack recall N/A. Needs a real attack holdout.
- Inherited `ISAA_ANN` (CICIDS2017 2017 traffic) misclassifies modern
  HTTPS/QUIC/STUN as DDoS/DoS — no retraining done.
- World-model / multi-step reports show macro-F1 0.21–0.25 from class absence,
  though binary attack-F1 0.85–0.91.
- Response module native Apply/Verify/Rollback disabled (capability conflict).
- **No `.github/` — no CI.** XDR slice + response module unstaged.
- Row-order-as-time proxy for the temporal pipeline (CICIDS2017 lacks usable
  timestamps) — heavily disclaimed; a real timestamped source would strengthen it.

## Feature transfer: network-attack-forecasting → NIDS

Ranked by value to the SIH submission. NAF paths are given relative to the
former repo root.

| # | Transfer | From (NAF path) | Rationale |
|---|---|---|---|
| 1 | **MLOps spine** — release gate, PSI drift detection, shadow retrain, autotrain daemon | `src/mlops/{release_gate,drift,shadow_retrain,scheduler}.py` | NIDS has none. USP #1 — "these people ship real systems". Port wholesale into `backend/`. |
| 2 | **4 dataset adapters** behind one state-schema contract | `src/data_pipeline/adapters/{ciciot2023,ctu13,darpa98,cicflowmeter,wireshark_csv}.py` | NIDS is CICIDS2017-only. Gives multi-dataset coverage + cross-dataset generalization evidence (closes the "single dataset" weakness). |
| 3 | **World-model regression head + `rollout()`** K-step autoregressive next-state forecast | `src/model/lstm.py`, `src/inference.py` (`rollout()`) | NIDS `backend/worldmodel/` exists but NAF's "predict the full next state vector, feed it back" design is the cleaner world-model story the PS asks for. Merge into `worldmodel/model.py` + `engine.py`. |
| 4 | **CI/CD + Docker compose + GHCR publish** | `.github/workflows/ci-cd.yml`, `Dockerfile`, `docker-compose.yml` | NIDS has **no `.github/`**. Port the pytest + build + publish workflow. |
| 5 | **STIX-lite alert export + SOC PDF report** | `src/explain/alert_export.py`, `src/soc_pdf.py` | NIDS has neither. Enterprise / CII framing the PS wants. |
| 6 | **Attention-weight explainability** ("which past window drove this") | `src/explain/attention.py` | Additive to NIDS's gradient×input saliency + SHAP service. |
| 7 | **Pure-Python CICFlowMeter-style flow reconstruction** + TLS/JA3 features | `src/data_pipeline/flow_reconstruction.py`, `src/explain/tls_features.py` | Removes NIDS's dependency on the externally-patched `cicflowmeter`. |
| 8 | **Weak MITRE 5-stage labeling rules** | `src/explainability/mitre.py`, `src/model/label_map.py` | NIDS's `backend/mitre/mapper.py` maps only 4 candidate techniques; NAF's substring→5-stage rules give the Recon…Exfiltration coverage the PS lists. |
| 9 | **Leakage-safe chronological-per-source split + admission control** | `src/data_pipeline/admission.py`, `src/model/dataset.py` | Stronger than NIDS's row-order proxy split. |
| 10 | The reference docs — `docs/{ARCHITECTURE,DATASETS,WORKFLOW,REFERENCES,SLIDES,DEMO_VIDEO_SCRIPT}.md` and `docs/SUBMISSION.md` | `docs/` | Ready-made submission scaffolding. |

### Keep from NIDS — do not regress

- Real live capture (`backend/capture/capture.py` — dumpcap/tcpdump + tshark table).
- Operator firewall RESPONSE module (`backend/response/`) — dry-run + SQLite audit. NAF only has review-only templates.
- 33-file test suite (vs NAF's 19).
- The already-redesigned no-build SOC UI (`frontend/`).

### Does not port cleanly

- NAF's **global per-time-window state** (CIC-IDS2018 released CSVs have no
  src/dst IP columns) vs NIDS's **per-flow** state + row-order-as-time proxy —
  two different state contracts. Pick one before merging the model code.
- NAF's Next.js dashboard — skip; NIDS's UI is further along.
