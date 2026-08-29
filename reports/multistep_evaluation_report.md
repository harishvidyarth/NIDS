# Multi-Step LSTM Evaluation Report

Direct H1-H6 row-order proxy evaluation. Horizons are windows, not validated elapsed seconds.

## Strict Holdout

- Train: Monday-WorkingHours.pcap_ISCX.csv, Tuesday-WorkingHours.pcap_ISCX.csv, Wednesday-workingHours.pcap_ISCX.csv, Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv, Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Validation: Friday-WorkingHours-Morning.pcap_ISCX.csv, Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Test: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

## Per-Horizon Results

| Horizon | Accuracy | Balanced accuracy | Macro-F1 | Attack F1 |
|---:|---:|---:|---:|---:|
| H1 | 0.4336 | 0.2468 | 0.2261 | 0.9117922520221371 |
| H2 | 0.4333 | 0.2466 | 0.2204 | 0.8853830733025363 |
| H3 | 0.4316 | 0.2457 | 0.2259 | 0.9119056707808456 |
| H4 | 0.4309 | 0.2453 | 0.2253 | 0.9092916595889247 |
| H5 | 0.4317 | 0.2457 | 0.2125 | 0.8451528659552434 |
| H6 | 0.4294 | 0.2445 | 0.2241 | 0.9045350123372755 |

## Limitations

- CICIDS2017 is an older, attack-clustered benchmark and does not represent current production traffic.
- ANN-derived four-state targets are model outputs, while raw CICIDS labels remain diagnostic metadata.
- Row order is a synthetic timing proxy; H1-H6 are windows and not validated seconds-ahead forecasts.
- Absent split/horizon classes are reported as N/A and are never fabricated or moved between splits.
- Longer horizons carry greater uncertainty; network-only evidence cannot confirm host intent or ATT&CK attribution.
