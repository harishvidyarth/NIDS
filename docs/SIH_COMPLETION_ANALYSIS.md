# SIH26153 — Completion %, Deviations, Feasibility, USP

> Compiled 2026-08-31 from a full read of the repo (source, tests, artifacts,
> training reports, docs) against `docs/PROBLEM_STATEMENT.md`. This is the
> source of truth for the "PS Completion Tracker" and "Deviations" tabs.

---

## 1. Feasibility verdict — does the PS "come into our hands"?

**Yes, strongly.** The repo already implements the reference architecture the PS
itself describes, end to end, on a laptop with no GPU and no cloud:

- CIC-IDS2018 ingestion -> 30s windowing -> weak MITRE labelling -> 3,273 state
  windows (`run_ingest.py`, `windows.parquet`).
- Dual-head LSTM: MITRE-stage softmax **plus a next-state regression head** —
  the regression head is what makes it a world model and not a relabelled
  classifier — with attention, `rollout()` K=5 autoregressive forward
  simulation (`src/model/`, `src/inference.py`, saved `lstm_weights.pt`).
- 4 dataset adapters behind one state-schema contract (CIC-IDS2018, CTU-13,
  DARPA-98, CICIoT2023).
- Attention + SHAP explainability, LogReg baseline, Streamlit + FastAPI +
  Next.js cockpit + headless CLI, 19 tests, CI/CD, Docker compose,
  `SUBMISSION.md` + rendered PDF, slide content, demo-video script.

The build risk is **not** "can we solve it" — it is **credibility under
evaluation**: the release gate is red, the packet-level path isn't trained, the
cross-dataset result is negative, and the repo is private with a large
uncommitted WIP branch. Those are the ~32% below, and they are fixable in the
~19-day window.

---

## 2. Completion % against the PS "Expected Solution"

Weighted estimate. Each row scored 0–100 on evidence in the repo today.

| # | PS requirement | Weight | State in repo (2026-08-31) | % |
|---|---|---:|---|---:|
| R1 | Feature pipeline — flow CSV **and** packet-level (PCAP, Scapy/PyShark) -> timestamped normalised matrix | 20 | CSV path solid (CIC-IDS2018, CTU-13, DARPA-98). `packet_features.py` + `tls_features.py` exist; a fused 33-dim flow+packet model is trained **only on DARPA-98**. The canonical v2 PCAP contract model is **not trained** — PCAP/live forecasting returns a readiness failure by design. | 60 |
| R2 | Trained world model (dynamics, not classifier) + scripts + weights + reproducible config | 25 | dual-head LSTM + regression head + `rollout()`; `train.py`, `run_train.py`, `config.yaml`; saved `lstm_weights.pt` (+ `.meta.json`), baseline, scaler, encoder. **`release_gate.passed = false`** (7 failures). | 75 |
| R3 | Supervised dynamics learning from attack-timeline annotations; **generalises to unseen attacks** | 15 | Trained from weak-labelled timelines. Cross-dataset (train CIC-IDS2018, test CTU-13): LSTM macro-F1 **0.51 < baseline 0.63** — an honest negative, currently unaddressed. | 45 |
| R4 | Infiltration engine — K-step sim -> prob time-series + predicted MITRE stage + top driving features | 15 | `rollout()` K=5, infiltration-probability timeline, per-window MITRE stage (**4 stages — no Reconnaissance**), attention + SHAP feature ranking. | 80 |
| R5 | Explainability per prediction (SHAP **or** attention), not black-box | 8 | Both: LSTM attention weights + `shap.LinearExplainer` on the LogReg proxy (documented trade-off — exact + fast on CPU). | 90 |
| R6 | Offline demo (Streamlit / Flask / CLI), PCAP **or** CSV, shows timeline + flagged flows + stage annotations, no cloud API | 10 | Streamlit (CSV upload, offline), FastAPI + Next.js cockpit, headless CLI. **PCAP input to the demo -> readiness failure** (v2 untrained). CSV path works. | 70 |
| R7 | Benchmark vs logistic-regression baseline: F1, precision, recall, FPR; measurable improvement | 7 | LogReg baseline trained on the same split; benchmark table in `docs/SUBMISSION.md` (LSTM F1 0.950 vs 0.894, FPR 0.018 vs 0.045 on a 651-seq held-out set). Needs reconciliation with the failing release gate before it's defensible. | 80 |

**Weighted core score:**
(20·60 + 25·75 + 15·45 + 15·80 + 8·90 + 10·70 + 7·80) / 100
= (1200 + 1875 + 675 + 1200 + 720 + 700 + 560) / 100
= **6930 / 100 ≈ 69%.**

### Evaluation-deliverable packaging (separate axis)

| Deliverable | State | % |
|---|---|---:|
| E1 Source code link | Repo is **private**; no public GitHub / Drive link yet | 40 |
| E2 README with setup | Present, detailed (`README.md`, `README-OPS.md`, `DEMO_RUNBOOK.md`) | 95 |
| E3 Architecture doc (max 2 pages) | `docs/ARCHITECTURE_NOTE.md` exists (judge-facing 1-page); needs final trim + single clean state story (D6) | 85 |
| E4 Demo video (max 2 min) | Script written (`DEMO_VIDEO_SCRIPT.md`); README claims a recording exists — **unverified**, treat as not done | 45 |
| E5 Technical presentation (max 5 slides) | Content written (`docs/SLIDES.md`); actual deck built externally — **unverified** | 55 |

Packaging weighted ≈ **64%**.

### Overall

**≈ 68% to a submission-ready state.** Core tech works; the remaining ~32% is
concentrated in exactly the places an evaluator inspects: a green quality gate,
the packet-level path, a generalization story that isn't a loss, a public code
link, and a real 2-minute video + 5-slide deck.

---

## 3. Deviation register (seeds the "Deviations" tab)

| ID | Deviation from the PS | Severity | Fix owner | Fix plan |
|---|---|---|---|---|
| **D1** | PS calls flow **and** packet features "required"; the trained working model is flow-only (CIC-IDS2018). PCAP/live path returns a readiness failure. | High | Jayakumar | Wire Scapy/PyShark packet features (TTL variance, TCP window size, IP fragment flags, retransmission count, port-scan signature) into a **trained** fused model on a public dataset with real PCAPs (DARPA-98 / CICIoT2023). Remove the readiness-failure stub from the demo path. |
| **D2** | PS wants the full 5-stage MITRE mapping incl. **Reconnaissance**; CIC-IDS2018 weak labels never surface a Recon class. | Medium | Jayakumar | Add a Recon weak-label rule keyed to port-scan signatures (sequential/randomised port access) on the flow path; verify a Recon window appears in the demo. |
| **D3** | PS wants generalisation to unseen attacks; CTU-13 cross-dataset test is a loss (LSTM < baseline). | Medium | Archana | Write it up honestly as a domain-shift section: report train-CIC / test-CTU + test-DARPA numbers, add a normalisation / feature-alignment pass, and frame remaining gap as scoped future work rather than hiding it. |
| **D4** | `release_gate.passed = false`: scenario-id leakage, per-class recall below threshold, one-step macro/micro below threshold. | High | Harish | Leakage-safe chronological-per-source splits, per-class validation-window minimums, class-weighted retrain; target a **passing** gate. Keep the failing report in-repo as evidence of rigor. |
| **D5** | Repo is private + ~67 uncommitted working-tree changes on a feature branch -> reproducibility / hygiene risk for judges. | Medium | Sujitha (+ Harish) | Commit and push the WIP, tag a release, make the repo public (or export a clean Drive zip) before submission. Record the visibility decision in the Milestones tab. |
| **D6** | Two divergent state contracts (78-feat legacy vs 33/79-feat canonical v2); the 2-page architecture doc must tell one story. | Low | Archana | Pick the contract that ships, describe only that one in `ARCHITECTURE_NOTE.md`, move the other to a "roadmap" line. |

---

## 4. Unique selling point (ranked — lead with the first three)

1. **Production-grade MLOps spine.** Release gate, PSI drift detection, shadow
   retrain, autotrain daemon, and honest negative-result reporting. Almost no
   competing SIH team on this PS has any of this — it is the clearest "these
   people ship real systems" signal for a judge.
2. **Four real dataset adapters** (CIC-IDS2018, CTU-13, DARPA-98, CICIoT2023)
   behind one state-schema contract, **plus published cross-dataset
   generalisation evidence.** Credibility over a cherry-picked single-dataset
   F1 that every other team will also quote.
3. **End-to-end defender workflow.** Forecast -> MITRE stage -> attention + SHAP
   evidence -> STIX-lite alert + one-page SOC PDF. Competitors stop at a raw
   dashboard; this carries an analyst to an artifact they can act on.
4. **Dual explainability** — attention weights (*which past window*) **and** SHAP
   (*which feature*), not one or the other.
5. **Fully offline, CPU-only, `docker compose up` one-command demo** — directly
   satisfies the PS's hard "no cloud API dependency" constraint that will trip
   up any team leaning on a hosted model.

---

## 5. What the numbers imply for the sprint

- The two **High** deviations (D1 packet path, D4 release gate) are ~40% of the
  missing core score between them. They are Phase 1 and they are the gate to
  everything downstream — do them first.
- D3 is cheap to *report* honestly and expensive to *fix* fully; report it in
  Phase 2, attempt a partial fix only if Phase 1 lands early.
- Packaging (E1/E4/E5) is not hard, just unstarted — Phase 3, owned by Sujitha.
- If Phases 1–2 finish with time to spare, the GNN state-transition variant is
  the single best "wow" add, because it is the differentiator competitors are
  reaching for and the PS explicitly lists it as an acceptable architecture.
