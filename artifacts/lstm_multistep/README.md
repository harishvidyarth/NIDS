# Phase 4 Direct Multi-Step Artifacts

`latest.json` points only to a candidate that passed finite-loss,
finite-probability, probability-sum, prediction-diversity,
validation-performance, and save/load gates. Each version contains the direct
H1–H6 Keras model, training-only scaler, ordered features/classes, dataset and
leakage manifests, frozen validation threshold, evaluation, benchmark, and
activation status.

The model predicts six horizons directly from observed `5 × 28` history. It
does not recursively feed four-class outputs into 28-feature inputs and does
not modify Phase 3 artifacts in `artifacts/lstm_forecaster/`.

All horizons are row-order proxy **windows**, not validated elapsed seconds.
CICIDS2017 is old and attack-clustered; the four-state targets are ANN outputs;
raw labels are diagnostic metadata; longer-horizon uncertainty and network-only
MITRE limitations remain material.
