# SIH26153 — Prior Art & Competitive Scan

> Compiled 2026-08-31. Purpose: know who else is solving this PS (or has solved
> the adjacent research problem) so the team's differentiation is deliberate,
> not accidental. Feeds the "Prior Art" tab of the tracker sheet.

## Method / coverage note

- **GitHub**, **arXiv / journals**, **USPTO**, and vendor sites were searched.
- **LinkedIn and Instagram** were searched for any named engineer, researcher,
  or industrialist publicly proposing a solution *to this specific PS*. Nothing
  attributable to a named individual was found there — the public activity is
  (a) student hackathon repos on GitHub and (b) prior academic work. State this
  plainly if a judge asks "has industry already done this."
- The commercial category that comes closest is **predictive threat
  intelligence** (signal correlation + risk scoring), which is *not* a learned
  network-state-transition simulation. No shipping product matches the PS's
  "world model P(S_t+1 | S_t) rollout" framing.

---

## A. Competing SIH 2026 teams on PS 26153 (direct competitors)

GitHub repos created/updated within days of the 21 Aug 2026 PS launch. Most are
converging on the exact reference architecture the PS itself suggests
(LSTM/GRU/Transformer/GNN world model over CIC-IDS2018 or CTU-13, K-step rollout,
MITRE ATT&CK stage mapping, SHAP/attention, offline Streamlit/Flask dashboard,
logistic-regression baseline).

| Team / repo | Approach | Link | Last seen | Maturity | Where we beat it |
|---|---|---|---|---|---|
| piyushkashyap160-spec — **CyberForecaster** | Temporal LSTM world model (23-dim, 5s windows, autoregressive K-step) + Temporal GNN variant + LogReg baseline. Claims F1 0.9826 / P 0.9692 / R 0.9965 on CSE-CIC-IDS2018; future-state RMSE 2.05 at K=1. 47 unit tests, dashboard. | <https://github.com/piyushkashyap160-spec/CyberForecaster> | 30 Aug 2026 | Prototype (research-leaning; small attack-sample caveats) | Their headline F1 is single-dataset. We publish **cross-dataset generalization** (CIC-IDS2018 -> CTU-13 / DARPA), an MLOps release gate, and drift detection they don't have. |
| SabarishR08 — **SIH26153-AI-Network-Attack-Forecasting** | Pipeline: traffic -> anomaly detection -> feature extraction -> ML forecasting -> kill chain -> dashboard. Reuses NTAV for anomaly detection; Flask dashboard; Docker/Render deploy. | <https://github.com/SabarishR08/SIH26153-AI-Network-Attack-Forecasting> | 30 Aug 2026 | Working prototype (34 commits, live monitoring demo) | Their forecasting is bolted onto a third-party anomaly detector. Ours is a **single trained world model** with a regression head that actually learns P(S_t+1 \| S_t), plus dual explainability. |
| HowSuyash — **AttackForecast** | World model P(S_t+1\|S_t) over 60s states from CTU-13 flow+packet telemetry; MITRE progression rollout; per-prediction explanations; offline CPU. | <https://github.com/HowSuyash/AttackForecast> | 25 Aug 2026 | Prototype | Very close in framing. We have **4 dataset adapters** behind one state schema and a fuller platform (FastAPI + Next.js cockpit + Streamlit + CLI). |
| raushankumarsah07 — repo named for the PS | PyTorch GRU world model for counterfactual "what-if" rollout; Express/React SOC dashboard; Solidity/Ethereum smart contracts for tamper-proof prediction auditing (leans into the "Blockchain & Cybersecurity" theme). | <https://github.com/raushankumarsah07/aI-based-network-attack-forecasting-from-network-traffic-data> | 26 Aug 2026 | Prototype / scaffold | Blockchain audit trail is a genuine differentiator for *them*. Our answer: STIX-lite alert + signed SOC PDF is lighter and demoable offline. Consider a tamper-evident prediction log as a stretch. |
| kunalsrivastava8810 — **Threat-Scope** | R Shiny SOC dashboard unifying CVE + malware + network intrusion data with ML attack prediction. | <https://github.com/kunalsrivastava8810/Threat-Scope> | 30 Aug 2026 | Prototype dashboard | Dashboard-first, model-thin. We are model-first with a reproducible training config and benchmark. |
| krishnashahane — **sentinel-sih** | "AI Network Attack Forecasting World Model (SIH26153, NTRO)". | <https://github.com/krishnashahane/sentinel-sih> | 22 Aug 2026 | Early | — |
| jeffreyfeoder — **SIH-prototype** ("TraceForge") | AI-based network attack forecasting prototype. | <https://github.com/jeffreyfeoder/SIH-prototype> | 27 Aug 2026 | Early prototype | — |
| mohdayaan99 — Network-Attack-Forecasting | AI-based forecasting on CIC-IDS2017. | <https://github.com/mohdayaan99/Network-Attack-Forecasting> | 27 Aug 2026 | Early | — |
| Others (early / no README): AnirudhRao-24, trinesh1666/cyber-ai-forecasting, ghantaakashchowdary, mubeenuddin03, Sharanya0307, adhithya123456433, Harsh33t/Sunniva-anti-ddos | Same PS, varying scope | GitHub | Aug 2026 | Idea / scaffold | — |

**Differentiation is happening on:** graph vs vector state, counterfactual
"what-if" rollouts, blockchain audit trail, and packet-level (PCAP) features vs
flow-only. Our packet-level path (D1) and generalization evidence (D3) put us on
the credible side of the last two.

---

## B. Research reference architectures & prior academic work

| Source | What it is | Link | Date | Relevance |
|---|---|---|---|---|
| cyber-laboratory / verpejas — **Visual Analytics Tool** | "A Visual Analytics Tool for Network Traffic Analysis Combining DTW Alignment and RNN-Based Attack Forecasting" — DTW alignment of traffic sequences + RNN forecasting + visual analytics front end. Predates the SIH launch. | <https://github.com/cyber-laboratory/Visual_Analytics_Tool> · <https://github.com/verpejas/Visual_Analytics_Tool> | Jul–Aug 2026 | Closest existing "reference architecture" for RNN traffic forecasting + a visual front end. Cite it; note our world-model regression head + MITRE mapping go beyond it. |
| arXiv 2508.18230 — **KillChainGraph** | ML framework predicting & mapping ATT&CK techniques; per-kill-chain-phase classifiers with inter-phase dependencies as a directed graph. | <https://arxiv.org/pdf/2508.18230> | Aug 2025 | Directly supports our R4 "map predicted behaviour to MITRE stages". Good citation for the stage-mapping design. |
| arXiv 2511.23000 — "A Modular Framework for Rapidly Building Intrusion Predictors" | Compose intrusion *predictors* (not detectors) from modular components. | <https://arxiv.org/pdf/2511.23000> | 2025 | Backs the "predict, don't classify" framing. |
| arXiv 2312.17270 — "Anticipated Network Surveillance" | Extrapolated study predicting cyber-attacks with ML + data analytics. | <https://arxiv.org/pdf/2312.17270> | Dec 2023 | Background citation. |
| Husák et al., *Computers & Security* (Elsevier) | "A comprehensive approach for network attack forecasting" — attack-graph + prediction; the canonical citation for this problem. | <https://dl.acm.org/doi/10.1016/j.cose.2015.11.005> | 2015 | Must-cite foundational reference. |
| *Technological Forecasting & Social Change* — "Forecasting Cyber Threats and Pertinent Mitigation Technologies" | Improved Bayesian Graph Neural Network forecasting attack/mitigation gaps; reproduced as GitHub `Mahi-0809/CyberForesight`. | <https://www.sciencedirect.com/science/article/pii/S0040162524006346> · <https://github.com/Mahi-0809/CyberForesight> | 2024 | Prior GNN-for-forecasting work; supports the GNN stretch item. |
| KTH-SSAS — **AttackSimSLR** | Systematic literature review of attack simulation / prediction methods. | <https://github.com/KTH-SSAS/AttackSimSLR> | ongoing | Bibliography to mine for the REFERENCES doc. |
| Ha & Schmidhuber — **World Models** (arXiv:1803.10122) | Conceptual basis: learn P(S_t+1 \| S_t), forward-simulate. | <https://arxiv.org/abs/1803.10122> | 2018 | The framing citation (already in `docs/REFERENCES.md`). |
| USPTO 12,368,739 — "Adaptive network attack prediction system" | Granted patent: adaptive system predicting network attacks. | <https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12368739> | granted 2025 | Freedom-to-operate / prior-art awareness. Our submission is open-source research, not a product — low risk, but know it exists. |

---

## C. Commercial / industry adjacency

| Vendor / source | What it does | Link | Gap vs the PS |
|---|---|---|---|
| SentinelOne — "AI-Powered Predictive Threat Intelligence" | Behavioural-signal correlation + predictive risk scoring. | <https://www.sentinelone.com/cybersecurity-101/threat-intelligence/predictive-threat-intelligence/> | Correlates signals; does not learn a network state-transition model or roll it forward K steps. |
| CyberProof — MITRE ATT&CK-based predictions | Predictive scoring keyed to ATT&CK. | <https://www.cyberproof.com/mitre-attck/top-7-cybersecurity-predictions-for-2025-based-on-mitre-attck-framework/> | Framework-driven forecasting, not a learned world model. |
| PatSnap / Eureka analyst note — "World Models and Cybersecurity" | States the most mature current use of world models in security is NIDS using RNN/Transformer models of normal traffic, with short (minutes–hours) horizons. | <https://eureka.patsnap.com/report-world-models-and-cybersecurity-improve-threat-prediction> | Confirms the space is early. Our short-horizon K-step rollout is squarely in the described state of the art. |

**Takeaway for the pitch:** there is no product to point at that does what the PS
asks; the academic building blocks exist and are cited; the real race is against
~two dozen other SIH teams building the same reference design. Win on rigor
(release gate, drift, generalization evidence) and on the end-to-end defender
workflow, not on a bigger single-dataset F1.
