# XDR Prototype Vertical Slice

This additive prototype connects the existing NIDS detector and short-horizon forecast to lightweight campaign context and operator-reviewed response drafting. It does not retrain or replace the ANN, LSTM, multi-step LSTM, or world-model paths.

## What was built

- `backend/ingest/`: offline Zeek conn/DNS/SSL JSON ingest, a 200-record synthetic sample, and DNS, TLS, beaconing, and byte-asymmetry enrichment.
- `backend/temporal/`: `STATE_FEATURE_NAMES_V2` appends six optional display/graph values after the stable 28 model inputs. Forecasting still reads only the first 28.
- `backend/graph/`: deterministic host/edge construction, benign-baseline learning, edge-surprise scoring, and a normalized campaign score.
- `backend/triage/`: deterministic advisory summaries and an optional explicitly configured local LLM endpoint. Triage has no response-module dependency.
- `backend/response/ladder.py`: `NONE → ALERT → RATE_LIMIT → FIREWALL_RULE → QUARANTINE` planning. Apply and rollback are dry-run audit events only; no subprocess is called.
- `backend/deception/`: a fake credential marker and a canary route whose hits become high-confidence triage evidence.
- Multi-step forecasts now expose `hazard_curve` and `time_to_attack_seconds`, derived with real 10-second windows.
- The `XDR / CAMPAIGN` dashboard tab renders enrichment, an inline SVG graph, triage, the dry-run ladder/audit, and deception hits.

## Configuration

All switches under `config/config.json` → `xdr` default to `false`: `enabled`, `ingest`, `enrich_windows`, `graph`, `triage`, `response_ladder`, `deception`, and `forecast_timing`. Set `enabled` and individual capabilities to `true` for normal local use. `xdr.llm.endpoint` is the only optional outbound destination; when unset, triage is deterministic and offline.

The demo uses a process-local `/api/xdr/demo/enable` override authorized by the same per-start loopback token as response operations. It does not rewrite configuration. Existing `/api/pipeline` output is unchanged when XDR is off.

## Run the demo

With the documented backend already running:

```bash
.venv-lstm312/bin/python scripts/xdr_demo.py
```

Or let the script start it with the existing Python 3.12 environment:

```bash
.venv-lstm312/bin/python scripts/xdr_demo.py --serve
```

Open `http://127.0.0.1:8765` and select `XDR / CAMPAIGN`. The script uploads the largest local feature sample, waits for prediction, ingests the synthetic Zeek sample, records a canary hit, prepares temporal windows, requests a multi-step forecast, builds the graph and triage, and records a response plan. The response command is a preview only.

## Safety and limitations

The XDR ladder never executes `pf`, `nftables`, `tc`, PowerShell, EDR, or directory-service commands. Targets are parsed as IP addresses, configured protected addresses are refused, containment above `RATE_LIMIT` requires operator acknowledgement, and TTL expiry appends an exact dry-run rollback event. A local response cannot mitigate a link-saturating DDoS; use ISP/CDN/upstream filtering.

## DEFERRED (not in this prototype)

- Real Zeek deployment and production log transport
- GNN training on NF-\*-v2 datasets
- Real firewall, EDR, or Active Directory enforcement
- Model retraining
- Foundation models
- Federated learning
