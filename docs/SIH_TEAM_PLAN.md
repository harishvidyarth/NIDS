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
| **Harish Vidyarth N** | ML / PyTorch; repo admin (`harishvidyarth`) | **Development** — model & forecast core | R2, R7, D4 (release gate); CI, branch protection, release tag; **E1 + D5** (repo public / Drive link); submits on the portal; mentors Manjari on the attack side |
| **Kaviya V** | AI prototype dev; **Team Lead** (`kaviya` branch) | **Development** — platform core + PM | R6; FastAPI backend + no-build SOC UI; `docker compose` one-command demo; STIX-lite alert + SOC PDF; **E2** (README + setup); owns Milestones tab + daily-report enforcement |
| **Jayakumar** | Linux networking (`jayakumar-hacker`) | **Prototype** — data & packet path | R1, D1 (packet/PCAP path), D2 (Recon label); wires the end-to-end demo prototype |
| **Archana D** | AI/DS, Power BI | **Prototype** — explainability & evaluation | R3, R4, R5, D3 (generalization write-up), D6; the 2-page architecture doc (E3); prototype output panels; drafts the portal submission text |
| **Manjari** | Reading/writing/analysis; learning offensive testing | **Pitch + Red-Team (in training)** | Delivers the 5-slide pitch + judge Q&A; runs attack scenarios against the prototype to produce unseen-attack evidence for **D3** (mentored by Harish + Jayakumar — see section 6) |
| **Sujitha** | Canva, PPT, UI/UX — **does not use git** | **Media & Design** | **E4** (2-min demo video, edited in Canva / CapCut) and **E5** (5-slide deck in Canva); SOC-UI mockups in Canva/Figma for a dev to implement; diagram polish. Hands final files to the team — never touches the repo. |

Every PS requirement (R1–R8) and every deviation (D1–D6) has exactly one owner.
R8 (enterprise / CII framing) is a shared narrative task led by Archana in the
architecture doc and reflected by Sujitha in the deck.

---

## 3. Milestone calendar

| Phase | Window | Goal | Lead(s) |
|---|---|---|---|
| 0 | Sep 1 | Tracker sheet live; PS saved; roles locked; commit + push the WIP; tag `v0.3-pre-sih` | Kaviya, Harish |
| 1 | Sep 1–6 | **D1** packet/PCAP path *trained* (not gated) + **D2** Recon label; **D4** release gate green | Jayakumar (D1/D2), Harish + Kaviya (D4) |
| 2 | Sep 7–11 | **D3** generalization write-up (honest domain-shift); final benchmark table (R7); 2-page architecture doc (E3, resolves **D6**); prototype wired end-to-end | Archana, Jayakumar |
| 3 | Sep 12–15 | 2-min demo video (E4 — Sujitha, Canva); 5-slide deck (E5 — Sujitha, Canva); `docker compose up` one-command demo on a clean machine; Manjari attack-training run | Sujitha (deck/video), Kaviya, Manjari |
| 4 | Sep 16–18 | Buffer + polish; judge Q&A prep; portal submission text (Archana drafts); **repo public / Drive code link** (E1, **D5** — Harish); pitch dry-run (Manjari) | Harish, Archana, Manjari |
| — | **Sep 19** | **Submit on the SIH portal** | Harish |

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

Out of scope for this sprint: `network-attack-forecasting` (retired — features
being ported per `docs/REPO_SWITCH.md`), real eBPF sensors, federated learning.
The dry-run firewall response module stays (demo-only, nothing auto-applied).

---

## 6. Teaching Manjari to attack (red-team, in training)

Goal: Manjari can, by Phase 3, run a repeatable set of attack scenarios against
the running prototype and read the output — this produces the "unseen attack"
evidence for **D3** and prepares her for judge Q&A.

**Mentors:** Harish (what the model sees / how to read a forecast) + Jayakumar
(the tooling / safe lab setup).

**Lab rules:** everything on `localhost` or a throwaway VM / an isolated lab
subnet the team controls. Never scan or flood any network, host, or service the
team does not own. Capture to pcap, replay into the prototype offline.

**Session 1 (~90 min, by Sep 8) — recon + scan:**
- `nmap -sS -T4 localhost` and `-sV`, `-p-` — watch a Reconnaissance window
  appear in the forecast (validates **D2**).
- `nmap` timing templates `-T2` vs `-T5` — see how a slow scan hides from
  flow thresholds but shows in packet-level timing features.
- Record each run as a pcap in `pcaps/redteam/`.

**Session 2 (~90 min, by Sep 11) — flood + replay:**
- `hping3 --flood -S -p 80 <lab-target>` (SYN flood) and a UDP flood — watch
  the infiltration-probability curve climb and the MITRE stage move to Impact.
- Replay a held-out attack family from CTU-13 / CICIoT2023 that was **not** in
  training, through the offline upload path — this is the D3 generalization run.
- She writes 1 paragraph per scenario: what she did, what the forecast showed,
  where it was wrong.

**Deliverable:** `docs/REDTEAM_SCENARIOS.md` — the commands, the pcaps, the
observed vs expected forecast per scenario. Feeds `docs/SIH_COMPLETION_ANALYSIS.md`
D3 and the judge Q&A doc.

**Tools:** `nmap`, `hping3` (or `scapy` for custom packets), `tcpdump` for
capture, the prototype's own file-upload replay path. All already installable on
the team's Linux boxes; Jayakumar sets up the lab VM.
