# LSTM Artifact Inventory

All model files are committed so the saved forecaster and its evaluation can be
reproduced offline. Paths in pointers and reports are repository-relative.

## Active artifact

`v1-3a5264a499ed` is the active Phase 3 artifact. It uses training protocol
`strict-final-train-scaler/v1`; its scaler is fitted only on the final training
windows. `latest.json` points to this version.

The independent terminal holdout contains BENIGN targets only. Its results
measure late-session benign stability, not attack forecasting. Attack-containing
rolling-origin results are internal model-selection diagnostics, not an untouched
test set. See `reports/lstm_evaluation_report.md` for the complete audit.

## Historical artifacts

`v1-6b89455037d9` and `v1-9f7373e0b3ad` predate the corrected scaler scope.
They are retained as historical defect evidence and for provenance only. Their
metrics must not be interpreted as normal generalization performance. The
explicit pre-fix evaluator output is preserved in
`reports/lstm_evaluation_report_pre_fix.json` and
`reports/lstm_evaluation_report_pre_fix.md`.

Each version includes its final Keras model, scaler, logistic baseline, class and
feature metadata, training report, source analysis, and rolling-origin selection
checkpoints. Runtime datasets and the fingerprinted window cache are intentionally
excluded from Git.
