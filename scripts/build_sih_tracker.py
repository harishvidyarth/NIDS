#!/usr/bin/env python3
"""Build (or refresh) the SIH26153 team tracker Google Sheet via the Sheets API.

Usage
-----
    # 1. one-time: install deps into the repo venv
    .venv/bin/pip install -r scripts/requirements-tracker.txt

    # 2. create the sheet (opens a browser once for Google consent)
    .venv/bin/python scripts/build_sih_tracker.py

    # offline: just write the CSV mirror, no network / no auth
    .venv/bin/python scripts/build_sih_tracker.py --csv-only

The OAuth client is the desktop ("installed") client_secret JSON already on this
machine. By default the script globs for it in ~/Downloads; override with
--client-secret. The user token is cached at scripts/.sih_tracker_token.json
(git-ignored) so consent is only needed once.

All seed content lives in the SHEETS constant below — one source of truth for
both the CSV mirror and the Sheets API payload. Edit there, re-run.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "sih_tracker_data"
TOKEN_PATH = SCRIPT_DIR / ".sih_tracker_token.json"
DEFAULT_CLIENT_SECRET_GLOB = os.path.expanduser(
    "~/Downloads/client_secret_*apps.googleusercontent.com.json"
)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
SPREADSHEET_TITLE = "SIH26153 — Team Tracker"

TEAM = ["Harish Vidyarth N", "Kaviya V", "Jayakumar", "Archana D", "Manjari", "Sujitha"]
STATUS_VALUES = ["Not started", "In progress", "Done", "Blocked"]
DEV_STATUS_VALUES = ["Open", "In progress", "Fixed", "Won't fix"]
SEVERITY_VALUES = ["High", "Medium", "Low"]

# ---------------------------------------------------------------------------
# Seed content. Each entry: tab title -> {"headers": [...], "rows": [[...], ...]}
# ---------------------------------------------------------------------------
SHEETS: dict[str, dict] = {
    "Overview": {
        "headers": ["Field", "Value"],
        "rows": [
            ["PS Number", "SIH26153"],
            ["Title", "AI based Network Attack Forecasting from Network Traffic Data"],
            ["Organisation", "National Technical Research Organisation (NTRO)"],
            ["Category / Theme", "Software / Blockchain & Cybersecurity"],
            ["Idea submission deadline", "2026-09-20"],
            ["Team working cutoff", "2026-09-19"],
            ["Days left to deadline", "=DATE(2026,9,20)-TODAY()"],
            ["Repo", "github.com/harishvidyarthcsecs/network-attack-forecasting (private)"],
            ["Public code link (E1)", "TBD — Sujitha"],
            ["Demo video link (E4)", "TBD — Sujitha"],
            ["Slides link (E5)", "TBD — Sujitha"],
            ["Tracker owner / PM", "Kaviya V"],
            ["Current core completion %", "='PS Completion'!B2"],
            ["USP 1", "Production-grade MLOps spine (release gate, drift, shadow retrain, honest negatives)"],
            ["USP 2", "4 real dataset adapters + published cross-dataset generalisation evidence"],
            ["USP 3", "End-to-end defender workflow: forecast -> MITRE stage -> attention+SHAP -> STIX-lite alert + SOC PDF"],
        ],
    },
    "PS Completion": {
        # B2 is the weighted rollup; put it high so Overview can point at it.
        "headers": ["Metric", "Value", "", "", "", "", "", ""],
        "rows": [
            ["Weighted core completion %",
             "=ROUND(SUMPRODUCT(D6:D12,E6:E12)/SUM(D6:D12),0)", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["Core requirements (R1-R7)", "", "", "", "", "", "", ""],
            ["ID", "Requirement", "Owner", "Weight", "% done", "Weighted",
             "Evidence", "Notes"],
            ["R1", "Feature pipeline: flow CSV + packet/PCAP -> normalised matrix",
             "Jayakumar", 20, 60, "=D6*E6/100", "windows.parquet; packet_features.py",
             "packet path trained only on DARPA-98; canonical v2 untrained (D1)"],
            ["R2", "Trained world model (dynamics, not classifier) + scripts + weights + config",
             "Harish", 25, 75, "=D7*E7/100", "lstm_weights.pt; train.py; config.yaml",
             "release_gate.passed=false (D4)"],
            ["R3", "Supervised dynamics learning; generalises to unseen attacks",
             "Archana", 15, 45, "=D8*E8/100", "CTU-13 cross-dataset run",
             "LSTM 0.51 < baseline 0.63 macro-F1 (D3)"],
            ["R4", "Infiltration engine: K-step sim -> prob timeline + MITRE stage + top features",
             "Archana", 15, 80, "=D9*E9/100", "rollout() K=5; mitre.py; attention+SHAP",
             "4 stages only, no Reconnaissance (D2)"],
            ["R5", "Explainability per prediction (SHAP or attention), not black-box",
             "Archana", 8, 90, "=D10*E10/100", "attention weights + shap.LinearExplainer",
             "SHAP on LogReg proxy (documented)"],
            ["R6", "Offline demo (Streamlit/Flask/CLI), PCAP or CSV, timeline+flags+stage",
             "Kaviya", 10, 70, "=D11*E11/100", "app/app.py; app/api.py; cli.py",
             "PCAP input -> readiness failure (D1)"],
            ["R7", "Benchmark vs logistic-regression baseline (F1/P/R/FPR)",
             "Harish", 7, 80, "=D12*E12/100", "docs/SUBMISSION.md benchmark table",
             "reconcile with failing release gate"],
            ["", "", "", "", "", "", "", ""],
            ["Deliverable packaging (E1-E5) — tracked, not in the core rollup", "", "", "", "", "", "", ""],
            ["E1", "Source code link (GitHub/Drive)", "Sujitha", "", 40, "",
             "repo private", "make public / Drive zip (D5)"],
            ["E2", "README with setup", "Sujitha", "", 95, "",
             "README.md; README-OPS.md; DEMO_RUNBOOK.md", "polish pass"],
            ["E3", "Architecture doc (max 2 pages)", "Archana", "", 85, "",
             "docs/ARCHITECTURE_NOTE.md", "trim to 2 pp; single state story (D6)"],
            ["E4", "Demo video (max 2 min)", "Sujitha", "", 45, "",
             "docs/DEMO_VIDEO_SCRIPT.md", "recording unverified"],
            ["E5", "Technical presentation (max 5 slides)", "Sujitha", "", 55, "",
             "docs/SLIDES.md", "deck unverified"],
        ],
    },
    "Team & Work Split": {
        "headers": ["Member", "Role", "Skills", "GitHub", "Owns (R# / D#)", "Contribution summary"],
        "rows": [
            ["Harish Vidyarth N", "Development — model & forecast core", "ML / PyTorch; repo admin",
             "harishvidyarth", "R2, R7, D4, D5, E1",
             "dual-head LSTM + K-step rollout; leakage-safe splits + per-class recall + retrain to a passing release gate (D4); LogReg benchmark (R7); CI, branch protection, release tag; make repo public / Drive code link (E1, D5); submit on the portal; mentors Manjari on the attack side"],
            ["Kaviya V", "Development — platform core + PM", "AI prototype development; Team Lead",
             "kaviya (branch)", "R6, E2",
             "FastAPI backend + no-build SOC UI; docker compose one-command demo; PCAP/CSV upload; STIX-lite alert + SOC PDF wiring (R6); README + setup instructions (E2); owns Milestones tab + daily-report enforcement"],
            ["Jayakumar", "Prototype — data & packet path", "Linux networking", "jayakumar-hacker",
             "R1, D1, D2",
             "CIC-IDS2018 + CTU-13 ingestion, 30s windowing, weak MITRE labels (R1); Scapy/PyShark packet features into a trained fused model (D1); Reconnaissance weak-label rule (D2); wires the end-to-end demo prototype"],
            ["Archana D", "Prototype — explainability & evaluation", "AI/DS, Power BI", "",
             "R3, R4, R5, D3, D6, E3",
             "attention + SHAP per prediction (R5); rolling-horizon eval + K-step engine outputs (R4); cross-dataset generalisation write-up (D3); benchmark visuals; 2-page architecture doc (E3, resolves D6); prototype output panels"],
            ["Manjari", "Pitch + Red-Team (in training)", "Reading/writing/analysis; learning offensive testing", "",
             "E5 delivery; feeds D3",
             "delivers the 5-slide pitch + judge Q&A; being taught to attack — runs nmap / hping3 / replayed-pcap scenarios against the prototype to produce unseen-attack evidence for D3 (mentored by Harish + Jayakumar); see docs/SIH_TEAM_PLAN.md section 6"],
            ["Sujitha", "Media & Design (no git)", "Canva, PPT, UI/UX", "",
             "E4 assets, E5 deck",
             "5-slide deck in Canva (from docs/SLIDES.md); 2-min demo video edit in Canva / CapCut (from docs/DEMO_VIDEO_SCRIPT.md); SOC-UI mockups in Canva/Figma for a dev to implement; diagram polish for the architecture doc. Hands final files to the team — does not touch the repo."],
        ],
    },
    "Milestones": {
        "headers": ["Phase", "Window", "Goal", "Owner", "Status", "Evidence link"],
        "rows": [
            ["0", "Sep 1", "Tracker sheet live; PS saved; roles locked; commit + push the WIP; tag v0.3-pre-sih", "Kaviya / Harish", "Not started", ""],
            ["1", "Sep 1-6", "D1 packet/PCAP path trained (not gated) + D2 Recon label; D4 release gate green", "Jayakumar (D1/D2) / Harish + Kaviya (D4, dev)", "Not started", ""],
            ["2", "Sep 7-11", "D3 generalisation write-up; final benchmark (R7); 2-page architecture doc (E3, resolves D6); prototype wired end-to-end", "Archana / Jayakumar", "Not started", ""],
            ["3", "Sep 12-15", "2-min demo video (E4, Sujitha in Canva); 5-slide deck (E5, Sujitha in Canva); docker compose one-command demo on a clean machine; Manjari attack-training run", "Sujitha (deck/video) / Kaviya / Manjari", "Not started", ""],
            ["4", "Sep 16-18", "Buffer + polish; judge Q&A prep; portal submission text (Archana drafts); repo public / Drive link (E1, D5, Harish); pitch dry-run (Manjari)", "Harish / Archana / Manjari", "Not started", ""],
            ["Submit", "Sep 19", "Submit on the SIH portal (1-day buffer before 20 Sep)", "Harish", "Not started", ""],
        ],
    },
    "Daily Reports": {
        "headers": ["Date", "Member", "Hours", "Done today", "Blockers", "Plan tomorrow", "Commit / PR links"],
        "rows": [],  # header-only; members add rows daily
    },
    "Deviations": {
        "headers": ["ID", "Description", "Severity", "Owner", "Fix plan", "Status", "Resolved date"],
        "rows": [
            ["D1", "PS calls flow AND packet features 'required'; working model is flow-only; PCAP/live path returns a readiness failure", "High", "Jayakumar", "Wire Scapy/PyShark packet features into a trained fused model on DARPA-98 / CICIoT2023; drop the readiness-failure stub from the demo path", "Open", ""],
            ["D2", "PS wants the full 5-stage MITRE map incl. Reconnaissance; CIC-IDS2018 weak labels never surface a Recon class", "Medium", "Jayakumar", "Add a Recon weak-label rule keyed to port-scan signatures on the flow path; verify a Recon window in the demo", "Open", ""],
            ["D3", "PS wants generalisation to unseen attacks; CTU-13 cross-dataset test is a loss (LSTM 0.51 < baseline 0.63 macro-F1)", "Medium", "Archana", "Honest domain-shift section: report train-CIC / test-CTU + test-DARPA; add feature-alignment pass; frame remaining gap as scoped future work", "Open", ""],
            ["D4", "release_gate.passed = false: scenario-id leakage, per-class recall below threshold, one-step macro/micro below threshold", "High", "Harish", "Leakage-safe chronological-per-source splits; per-class validation-window minimums; class-weighted retrain; target a passing gate; keep the failing report as rigor evidence", "Open", ""],
            ["D5", "Repo hygiene: uncommitted WIP + XDR/response slice unstaged -> reproducibility risk for judges", "Medium", "Harish", "Commit + push the WIP, tag a release, make the repo public (or export a clean Drive zip) before submission", "Open", ""],
            ["D6", "Two divergent state contracts (78-feat legacy vs 33/79-feat canonical v2); the 2-page doc must tell one story", "Low", "Archana", "Pick the contract that ships, describe only that one in ARCHITECTURE_NOTE.md, move the other to a roadmap line", "Open", ""],
        ],
    },
    "Prior Art": {
        "headers": ["Name", "Link", "Approach", "Maturity", "What we do better"],
        "rows": [
            ["CyberForecaster (piyushkashyap160-spec)", "https://github.com/piyushkashyap160-spec/CyberForecaster", "Temporal LSTM world model + Temporal GNN + LogReg baseline; claims F1 0.9826 on CSE-CIC-IDS2018", "Prototype", "Their F1 is single-dataset; we publish cross-dataset generalisation + a release gate + drift detection"],
            ["SIH26153-AI-Network-Attack-Forecasting (SabarishR08)", "https://github.com/SabarishR08/SIH26153-AI-Network-Attack-Forecasting", "traffic -> anomaly detection -> ML forecasting -> kill chain -> Flask dashboard; reuses NTAV", "Working prototype (34 commits)", "Forecasting bolted onto a 3rd-party detector; ours is one trained world model with a regression head + dual explainability"],
            ["AttackForecast (HowSuyash)", "https://github.com/HowSuyash/AttackForecast", "World model P(S_t+1|S_t) over 60s CTU-13 states; MITRE rollout; offline CPU", "Prototype", "We have 4 dataset adapters behind one schema + a fuller platform (FastAPI + Next.js + Streamlit + CLI)"],
            ["raushankumarsah07 (PS-named repo)", "https://github.com/raushankumarsah07/aI-based-network-attack-forecasting-from-network-traffic-data", "PyTorch GRU world model, counterfactual what-if rollout, Express/React SOC dashboard, Solidity smart-contract audit trail", "Prototype / scaffold", "Their blockchain audit trail is real differentiation; our answer is a lighter offline hash-chained prediction log (stretch item)"],
            ["Threat-Scope (kunalsrivastava8810)", "https://github.com/kunalsrivastava8810/Threat-Scope", "R Shiny SOC dashboard unifying CVE + malware + intrusion data with ML prediction", "Prototype dashboard", "Dashboard-first, model-thin; we are model-first with reproducible training config + benchmark"],
            ["Visual Analytics Tool (cyber-laboratory / verpejas)", "https://github.com/cyber-laboratory/Visual_Analytics_Tool", "DTW alignment of traffic sequences + RNN-based attack forecasting + visual analytics UI (research, predates SIH)", "Research tool / paper artifact", "Closest reference architecture; our world-model regression head + MITRE stage mapping go beyond it"],
            ["KillChainGraph (arXiv 2508.18230)", "https://arxiv.org/pdf/2508.18230", "Per-kill-chain-phase classifiers with inter-phase dependencies as a directed graph", "Preprint + code", "Reference for our stage-mapping design; cite, don't compete"],
            ["Husak et al. 2015, Computers & Security", "https://dl.acm.org/doi/10.1016/j.cose.2015.11.005", "Attack-graph + prediction; canonical network attack forecasting paper", "Published, widely cited", "Foundational citation"],
            ["SentinelOne / CyberProof predictive threat intel", "https://www.sentinelone.com/cybersecurity-101/threat-intelligence/predictive-threat-intelligence/", "Behavioural-signal correlation + predictive risk scoring (commercial)", "Shipping products", "They correlate signals; they do not learn a network state-transition model or roll it forward K steps"],
        ],
    },
    "Risks": {
        "headers": ["Risk", "Impact", "Likelihood", "Mitigation", "Owner"],
        "rows": [
            ["D1 or D4 slips past Sep 6", "High — cascades into every later phase", "Medium", "Cut the GNN stretch + non-blocking polish first; daily check at standup", "Kaviya"],
            ["Manjari still learning offensive testing at demo time", "Medium — weak D3 attack evidence + shaky Q&A", "Medium", "Structured attack-training plan in docs/SIH_TEAM_PLAN.md section 6; Harish + Jayakumar pair with her twice before Phase 3; canned scenario scripts committed", "Harish"],
            ["Release gate can't be made green in time", "High — credibility hit under evaluation", "Medium", "Ship with the honest failing report + a documented remediation plan rather than hiding it", "Harish"],
            ["Cross-dataset generalisation stays a net loss", "Medium — weakens R3 claim", "High", "Report honestly as domain shift + scoped future work; do not overclaim in the deck", "Archana"],
            ["Demo video / deck started too late", "Medium — rushed E4/E5", "Medium", "Hard-schedule Phase 3 Sep 12-15; script + slide content already written", "Sujitha"],
            ["Repo stays private at submission", "High — E1 fails outright", "Low", "Phase 4 checklist item with Harish as admin; decision recorded in Milestones tab", "Sujitha"],
            ["Two dozen teams ship the same reference design", "Medium — blends in", "High", "Lead the pitch with the MLOps spine + generalisation evidence, not a bigger single-dataset F1", "Kaviya"],
        ],
    },
}

TAB_ORDER = list(SHEETS.keys())


def write_csvs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for title, spec in SHEETS.items():
        fname = title.lower().replace(" & ", "_").replace(" ", "_") + ".csv"
        path = CSV_DIR / fname
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(spec["headers"])
            w.writerows(spec["rows"])
        print(f"  wrote {path.relative_to(SCRIPT_DIR.parent)}")


def resolve_client_secret(cli_value: str | None) -> str:
    if cli_value:
        if not os.path.exists(cli_value):
            sys.exit(f"client secret not found: {cli_value}")
        return cli_value
    matches = sorted(glob.glob(DEFAULT_CLIENT_SECRET_GLOB))
    if not matches:
        sys.exit(
            "no client_secret_*.json found in ~/Downloads; pass --client-secret PATH"
        )
    if len(matches) > 1:
        print(f"  multiple client secrets found, using: {matches[0]}")
    return matches[0]


def get_credentials(client_secret_path: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        print(f"  token cached at {TOKEN_PATH.relative_to(SCRIPT_DIR.parent)}")
    return creds


def a1_col(n: int) -> str:
    s = ""
    while n >= 0:
        s = chr(n % 26 + 65) + s
        n = n // 26 - 1
    return s


def build_requests(sheet_id_by_title: dict[str, int]) -> list[dict]:
    """Formatting + data-validation requests applied after values are written."""
    reqs: list[dict] = []
    for title, sid in sheet_id_by_title.items():
        # freeze + bold header row
        reqs.append({
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }
        })
        reqs.append({
            "repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1},
                "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                "fields": "userEnteredFormat.textFormat.bold",
            }
        })
        reqs.append({
            "updateDimensionProperties": {
                "range": {"sheetId": sid, "dimension": "COLUMNS",
                          "startIndex": 0, "endIndex": 8},
                "properties": {"pixelSize": 260},
                "fields": "pixelSize",
            }
        })

    def validation(title: str, col: int, values: list[str], start_row: int = 1,
                   end_row: int = 2000) -> dict:
        return {
            "setDataValidation": {
                "range": {"sheetId": sheet_id_by_title[title], "startRowIndex": start_row,
                          "endRowIndex": end_row, "startColumnIndex": col,
                          "endColumnIndex": col + 1},
                "rule": {
                    "condition": {"type": "ONE_OF_LIST",
                                  "values": [{"userEnteredValue": v} for v in values]},
                    "showCustomUi": True, "strict": False,
                },
            }
        }

    reqs.append(validation("Daily Reports", 1, TEAM))
    reqs.append(validation("Milestones", 4, STATUS_VALUES))
    reqs.append(validation("Deviations", 2, SEVERITY_VALUES))
    reqs.append(validation("Deviations", 5, DEV_STATUS_VALUES))
    return reqs


def _write_all_tabs(svc, ssid: str, sheet_id_by_title: dict[str, int]) -> None:
    data = []
    for title, spec in SHEETS.items():
        values = [spec["headers"]] + [[str(c) for c in row] for row in spec["rows"]]
        end_col = a1_col(max(len(spec["headers"]), 1) - 1)
        data.append({"range": f"'{title}'!A1:{end_col}{len(values)}", "values": values})
    svc.spreadsheets().values().batchUpdate(
        spreadsheetId=ssid,
        body={"valueInputOption": "USER_ENTERED", "data": data},
    ).execute()
    svc.spreadsheets().batchUpdate(
        spreadsheetId=ssid, body={"requests": build_requests(sheet_id_by_title)},
    ).execute()


def push_to_sheets(client_secret_path: str) -> str:
    from googleapiclient.discovery import build

    creds = get_credentials(client_secret_path)
    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

    created = svc.spreadsheets().create(body={
        "properties": {"title": SPREADSHEET_TITLE},
        "sheets": [{"properties": {"title": t, "index": i}}
                   for i, t in enumerate(TAB_ORDER)],
    }).execute()
    ssid = created["spreadsheetId"]
    sheet_id_by_title = {s["properties"]["title"]: s["properties"]["sheetId"]
                         for s in created["sheets"]}
    _write_all_tabs(svc, ssid, sheet_id_by_title)
    return created["spreadsheetUrl"]


def update_sheets(client_secret_path: str, spreadsheet_id: str) -> str:
    """Rewrite every tab's values + validation on an EXISTING spreadsheet.

    Adds any missing tab; clears A:Z of each existing tab first so shrunk
    tables don't leave stale rows. Does not touch Daily Reports rows below the
    header (its seed is header-only, and the clear+rewrite keeps row 1 only —
    so run updates before the team starts logging, or expect their rows wiped).
    """
    from googleapiclient.discovery import build

    creds = get_credentials(client_secret_path)
    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

    meta = svc.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id_by_title = {s["properties"]["title"]: s["properties"]["sheetId"]
                         for s in meta["sheets"]}
    add_reqs = [{"addSheet": {"properties": {"title": t}}}
                for t in TAB_ORDER if t not in sheet_id_by_title]
    if add_reqs:
        res = svc.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": add_reqs}).execute()
        for r in res.get("replies", []):
            p = r["addSheet"]["properties"]
            sheet_id_by_title[p["title"]] = p["sheetId"]

    svc.spreadsheets().values().batchClear(
        spreadsheetId=spreadsheet_id,
        body={"ranges": [f"'{t}'!A:Z" for t in SHEETS]},
    ).execute()
    _write_all_tabs(svc, spreadsheet_id, sheet_id_by_title)
    return meta["spreadsheetUrl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-only", action="store_true",
                    help="write the CSV mirror only; no network, no auth")
    ap.add_argument("--client-secret", default=None,
                    help="path to the OAuth desktop client_secret JSON "
                         "(default: glob ~/Downloads/client_secret_*apps.googleusercontent.com.json)")
    ap.add_argument("--update", metavar="SPREADSHEET_ID", default=None,
                    help="rewrite every tab on this EXISTING spreadsheet instead "
                         "of creating a new one (clears A:Z per tab first)")
    args = ap.parse_args()

    print("Writing CSV mirror ...")
    write_csvs()

    if args.csv_only:
        print(f"\nCSV-only mode. Import each file in {CSV_DIR} via "
              "Google Sheets > File > Import.")
        return

    try:
        secret = resolve_client_secret(args.client_secret)
        print(f"Using client secret: {secret}")
        if args.update:
            url = update_sheets(secret, args.update)
            print(f"\nUPDATED existing tracker sheet: {url}")
        else:
            url = push_to_sheets(secret)
            print(f"\nDONE. Tracker sheet: {url}")
            print("Share it with the team (edit access) and pin the tab order.")
    except ImportError:
        sys.exit("Missing deps. Run: "
                 ".venv/bin/pip install -r scripts/requirements-tracker.txt")


if __name__ == "__main__":
    main()
