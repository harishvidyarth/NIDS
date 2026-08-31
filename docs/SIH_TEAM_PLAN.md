# SIH26153 — Team Plan, Deadlines, Daily Reports

> Compiled 2026-08-31. Anchor deadline: **SIH portal idea submission,
> 20 September 2026.** Build scope: this repo only
> (`network-attack-forecasting`) — fix the gaps in
> `docs/SIH_COMPLETION_ANALYSIS.md`. Live tracking is in the Google Sheet built
> by `scripts/build_sih_tracker.py`.

---

## 1. Important dates (SIH 2026)

| Date | Milestone | Source |
|---|---|---|
| 21 Aug 2026 | PS list launched; team nomination + idea submission opened | reskilll.com, vssut.ac.in |
| **20 Sep 2026** | **Idea / PS submission deadline on the SIH portal** — our hard cutoff | SIH 2026 PS data field (per-PS) |
| ~Sep 2026 | College internal / SPOC hackathons (each college sets its own date; a peer institute used 08 Sep for PPT submission) | internshala.com, vjit.edu.in |
| Oct–Nov 2026 | Online evaluation / national shortlisting | reskilll.com, techpathdaily.com |
| Dec 2026 | Grand Finale (36-hour, at nodal centres) | reskilll.com |

**Working cutoff: submit on 19 Sep 2026** (1 clear day before the portal
closes). If our college sets an earlier internal date, Kaviya updates the
Milestones tab and the calendar below compresses to hit it first.

---

## 2. Team & role split

| Member | Skill | Role | Owns (PS req + deviations) |
|---|---|---|---|
| **Jayakumar** | Linux networking (`jayakumar-hacker`) | Data & Features Lead | R1, D1 (packet/PCAP path), D2 (Recon label) |
| **Harish Vidyarth N** | ML / PyTorch; repo admin (`harishvidyarthcsecs`) | Model & Forecast Lead | R2, R7, D4 (release gate); CI, branch protection, release tag |
| **Archana D** | AI/DS, Power BI | Explainability & Evaluation Lead | R3, R4, R5, D3 (generalization write-up), D6; the 2-page architecture doc (E3) |
| **Kaviya V** | AI prototype dev; **Team Lead** (`kaviya` branch) | Platform & Demo Lead + PM | R6; `docker compose` one-command demo; STIX-lite alert + SOC PDF; owns Milestones tab + daily-report enforcement |
| **Manjari** | Analysis, testing, offensive testing | QA & Red-Team Lead | Fresh-machine install test; adversarial / unseen-attack testing feeding D3; pytest coverage on the new packet path; judge Q&A doc |
| **Sujitha** | PPT, frontend UI/UX | Media & Submission Lead | E1 (repo public / Drive link, D5), E2 (README polish), E4 (2-min video), E5 (5-slide deck); cockpit UI/UX polish |

Every PS requirement (R1–R8) and every deviation (D1–D6) has exactly one owner.
R8 (enterprise / CII framing) is a shared narrative task led by Archana in the
architecture doc and Sujitha in the deck.

---

## 3. Milestone calendar

| Phase | Window | Goal | Lead(s) |
|---|---|---|---|
| 0 | Sep 1 | Tracker sheet live; PS saved; roles locked; commit + push the ~67-file WIP; tag `v0.3-pre-sih` | Kaviya, Sujitha, Harish |
| 1 | Sep 1–6 | **D1** packet/PCAP path *trained* (not gated) + **D2** Recon label; **D4** release gate green | Jayakumar, Harish |
| 2 | Sep 7–11 | **D3** generalization write-up (train CIC → test CTU-13 / DARPA, honest); final benchmark table (R7); 2-page architecture doc (E3, resolves **D6**) | Archana (+ Jayakumar, Harish) |
| 3 | Sep 12–15 | 2-min demo video (E4); 5-slide deck (E5); `docker compose up` one-command demo verified on a clean machine; UI/UX polish | Sujitha, Kaviya, Manjari |
| 4 | Sep 16–18 | Buffer + polish; judge Q&A doc; SIH portal submission text; **repo public / Drive code link** (E1, **D5**) | Sujitha, Manjari, all |
| — | **Sep 19** | **Submit on the SIH portal** | Sujitha (Harish confirms) |

Dependency: Phases 2–4 assume Phase 1 lands on time. If D1 or D4 slips past
Sep 6, Kaviya cuts the GNN stretch and any non-blocking polish first.

---

## 4. Daily report protocol

- **Where:** the "Daily Reports" tab of the SIH tracker sheet.
- **When:** every member fills their row by **21:00 IST every working day**.
- **Row:** Date · Member · Hours · Done today · Blockers · Plan tomorrow ·
  Commit/PR links.
- **One row per member per day.** Use the Member dropdown; don't rename the
  column headers.
- **Enforcement:** Kaviya (PM) checks completeness at the 09:30 IST standup.
  An empty row from the previous day = that person is a blocker until they post.
- **Standup:** 10 min, 09:30 IST, async-friendly — read the sheet first, discuss
  only blockers.

---

## 5. "What else can we build on it" — backlog (ranked)

Each item both closes a deviation and strengthens a USP.

1. **Packet-level PCAP ingestion** via Scapy/PyShark into the *same* state schema
   (TTL variance, TCP window size, IP fragment flags, retransmission count,
   port-scan signature), trained and un-gated. — closes **D1**, satisfies R1.
2. **Recon weak-label rule** from port-scan signatures on the flow path. —
   closes **D2**, completes the 5-stage MITRE map (R4).
3. **Leakage-safe retrain to a passing release gate**, keeping the failing
   report in-repo as rigor evidence. — closes **D4**, makes R7 defensible.
4. **Cross-dataset generalization section**: train CIC-IDS2018 → test CTU-13 /
   DARPA, reported honestly with a domain-shift note. — closes **D3**, USP #2.
5. **STIX-lite alert + one-page SOC PDF** per forecast — already scaffolded
   (`src/soc_pdf.py`), finish wiring into the demo. — USP #3, R8 framing.
6. **GNN state-transition variant** as a "future work / stretch" branch — only
   if Phases 1–2 finish early. PS lists a GNN as an acceptable architecture and
   it is the differentiator competitors are reaching for.
7. **Tamper-evident prediction log** (hash-chained JSONL) — a lightweight
   offline answer to competitors' blockchain audit trail; nice-to-have.

Out of scope for this sprint: the sibling `NIDS` repo, live firewall response,
real eBPF sensors, federated learning.
