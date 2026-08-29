# LSTM Evaluation Report

## Model Identity

- Model: `artifacts/lstm_forecaster/v1-3a5264a499ed/model.keras`
- Model SHA-256: `c8f1f98ed01576ffc0c6490bdd539828c0001b1ed53e5a4182c11e1fa9262f43`
- Scaler: `artifacts/lstm_forecaster/v1-3a5264a499ed/scaler.bin`
- Features: 28
- Sequence length: 5
- Classes: BENIGN, DDoS, DoS, PortScan

## Timestamp Methodology

- REAL TIMESTAMP TEMPORAL DATA: not used by this saved LSTM artifact.
- ROW-ORDER / SYNTHETIC-TIMESTAMP TEMPORAL PROXY: one source row is treated as one proxy second and aggregated into nominal 10-second windows.
- This is an experimental temporal-ordering proxy, not evidence of real-world time-to-attack forecasting.

## Temporal Leakage Audit

- Session Boundary Leakage: **PASS**
- Train Validation Overlap: **PASS**
- Train Test Overlap: **PASS**
- Scaler Train Only: **PASS**
- Next Window Alignment: **PASS**
- Future Feature Leakage: **PASS**
- Duplicate Sequence Leakage: **PASS**
- Invalid Features Exclusion: **PASS**
- Synthetic Session Continuity: **PASS**

Overall: **PASS**

## Dataset Distribution

### Train

- X: `(149698, 5, 28)`
- y: `(149698,)`
- BENIGN: 120130
- DDoS: 0
- DoS: 23149
- PortScan: 6419

### Validation

- X: `(16618, 5, 28)`
- y: `(16618,)`
- BENIGN: 16499
- DDoS: 0
- DoS: 0
- PortScan: 119

### Test

- X: `(29339, 5, 28)`
- y: `(29339,)`
- BENIGN: 29339
- DDoS: 0
- DoS: 0
- PortScan: 0

## Original Chronological Evaluation

- Accuracy: 0.999659
- Balanced accuracy: 0.999659
- Macro F1: 0.249957
- Weighted F1: 0.999830
- Attack recall: N/A — class absent from evaluation set
- Attack F1: N/A — class absent from evaluation set
- Absent classes: DDoS, DoS, PortScan

## Attack-Containing Evaluation

- Status: NOT_AVAILABLE
- Existing dataset cannot provide a sufficiently representative attack-containing chronological holdout.

## Transition Analysis

- BENIGN -> BENIGN: n=16492, accuracy=0.998545, mean true-state probability=0.984801
- PortScan -> PortScan: n=112, accuracy=0.955357, mean true-state probability=0.913265
- BENIGN -> PortScan: n=7, accuracy=0.571429, mean true-state probability=0.468471 (low support)
- PortScan -> BENIGN: n=7, accuracy=0.000000, mean true-state probability=0.040113 (low support)

## Rolling-Origin Diagnostics

- Status: INTERNAL_VALIDATION_ONLY
- Fold 1: samples=33251, attacks=9385, macro F1=0.717701, attack F1=0.936723
- Fold 2: samples=33251, attacks=2952, macro F1=0.493379, attack F1=0.975602
- Fold 3: samples=33253, attacks=1483, macro F1=0.489561, attack F1=0.960133

## Baseline Comparison — Test

| Metric | Logistic Regression | LSTM |
|---|---:|---:|
| Accuracy | 0.999966 | 0.999659 |
| Balanced Accuracy | 0.999966 | 0.999659 |
| Macro F1 | 0.249996 | 0.249957 |
| Weighted F1 | 0.999983 | 0.999830 |
| Attack Recall | N/A — class absent from evaluation set | N/A — class absent from evaluation set |
| Attack F1 | N/A — class absent from evaluation set | N/A — class absent from evaluation set |

## Limitations

- CICIDS2017 is an aging benchmark and does not represent every modern network or threat domain.
- DDoS is absent from all configured source sessions; the model cannot be validated for that class here.
- The strict test split contains only BENIGN targets, so attack recall/F1 are undefined and multiclass AUC is not valid.
- Network-flow evidence cannot confirm ATT&CK techniques, host/process behavior, or adversary intent.
- Mapping confidence is deterministic and heuristic, not statistically calibrated.

## MITRE ATT&CK Context

- ATT&CK version: 19.1
- Data modified: 2026-05-12T14:00:00.188Z
- Offline metadata: `backend/mitre/data/enterprise_attack_v19_1_subset.json`
- Source: https://github.com/mitre-attack/attack-stix-data/blob/master/enterprise-attack/enterprise-attack-19.1.json
- Implemented candidates: T1046 Network Service Discovery, T1498 Network Denial of Service, T1499 Endpoint Denial of Service, T1595 Active Scanning

## Conclusion

The saved LSTM was loaded and evaluated without initialization or pre-evaluation training. The leakage audit passed, but the independent test set is BENIGN-only; it measures late-session benign stability, not attack forecasting. The existing sessions cannot provide a defensible independent attack-containing chronological holdout.
