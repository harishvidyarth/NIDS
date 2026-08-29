# Existing Pipeline Audit (as of clone, main @ 39316cb)

## Repository contents (everything that exists — nothing else)

```
ANN Model/
  ISAA_ANN.h5      412K  trained Keras ANN (sigmoid MLP, softmax 4-class output)
  minmax.bin        4K   joblib-pickled sklearn MinMaxScaler fit on the 77 training features
NIDS (Prediction).ipynb   inference notebook (loads model+CSV, predicts, prints result)
NIDS - ANN.ipynb          training notebook for the ANN above
NIDS - ML.ipynb           training notebook for RandomForest / DecisionTree / KNN
README.md
Report.pdf
packetsniff.sh            tcpdump wrapper (capture only)
```

There is **no** CICFlowMeter code, **no** backend, **no** frontend/UI, **no** config
files, **no** requirements.txt/environment file, and **no** saved Random
Forest/Decision Tree/KNN model artifact anywhere in the repo. These are the
findings the rest of this doc explains.

## 1. How PCAP capture works

`packetsniff.sh` is a thin wrapper around `tcpdump`:

```
./packetsniff.sh <SourceIP> <DestinationIP> <output.pcap> [interface=eth0]
sudo tcpdump -w <output.pcap> -U -i <interface> -v src <SourceIP> and dst <DestinationIP>
```

It requires `sudo`/root, a Linux host with `tcpdump` installed, and both a
source and destination IP filter (no "capture everything on this interface"
mode). It runs until the user hits Ctrl+C — there's no duration flag, no
packet-count flag, and no programmatic stop/status/return value. It is
Linux-only (`sudo`, `eth0` default); it will not run as-is on Windows.

## 2. How CICFlowMeter is invoked

**It isn't, anywhere in this repo.** The README describes the intended step
("This PCAP file will be fed into CICFlowMeter...") but there is no script,
config, or notebook cell that calls CICFlowMeter. In practice the original
author ran the CICFlowMeter GUI/CLI tool by hand, outside this repo, and
manually copied the resulting CSV to a local path
(`C:\Users\varun\Downloads\test3_ISCX.csv`) that is hardcoded in the
prediction notebook. CICFlowMeter itself is a separate third-party Java
project (ISCX/CICFlowMeter) that must be obtained and built independently —
it is not vendored or referenced by commit/URL in this repo.

## 3. Where the extracted CSV is stored

Nowhere standardized. The prediction notebook reads a single hardcoded
absolute path on the original author's machine. There is no `features/` or
similar output directory convention in the repo.

## 4. Which features are consumed by the model

The ANN was trained (`NIDS - ANN.ipynb`) on the classic CICIDS2017 78-column
flow CSV, minus `Fwd Header Length.1` (duplicate column) and `Label`, i.e.
**77 numeric flow features** (Destination Port, Flow Duration, all the
Fwd/Bwd packet/byte/IAT/flag/active/idle statistics — the standard
CICFlowMeter feature set). All 77 columns are scaled with a single
`MinMaxScaler` (`minmax.bin`) fit on the training data before being fed to
the model.

The prediction notebook (`NIDS (Prediction).ipynb`), run against a raw
CICFlowMeter CSV, drops `Flow ID, Src IP, Src Port, Dst IP, Protocol,
Timestamp, Label` to arrive at the same 77 columns, then casts to
`float32`.

## 5. Which model is currently used

Only the **ANN** (`ANN Model/ISAA_ANN.h5`) is actually deployable — it's the
only model that was ever saved to disk. It's a plain `Sequential` Keras MLP:
`128 → 128 → 64 → 64 → 32 → 32 → 16 → 16 → 4`, all `sigmoid` hidden
activations, `softmax` output, trained with `categorical_crossentropy` /
`rmsprop`. Test accuracy in the training notebook was ~93%.

Random Forest / Decision Tree / KNN are trained and scored inside
`NIDS - ML.ipynb` for comparison, but **none of them are ever
`joblib.dump`'d or otherwise persisted** — that notebook only prints
accuracy numbers. There is no `.pkl`/`.joblib` file for any classical model
in the repo, so "Random Forest" is not currently usable for inference,
despite being mentioned in the README/report as one of the compared models.

## 6. What the model predicts

4-class softmax over `['BENIGN', 'DDoS', 'DoS', 'PortScan']` (alphabetical
order from `pd.get_dummies`, confirmed by the class-name printout in the
prediction notebook: 0=Benign, 1=DDoS, 2=DoS, 3=PortScan). It is a
per-flow, single-snapshot classification — not a time-series/forecast.

## 7. How inference is triggered

Manually, by opening `NIDS (Prediction).ipynb` in Jupyter, editing the two
hardcoded paths (model `.h5` path and input CSV path — both point to the
original author's machine and do not exist elsewhere), and running all
cells top to bottom. There is no script, CLI entry point, or API — nothing
callable from outside the notebook.

## 8. Missing dependencies / configuration / bugs found

- **No dependency manifest** anywhere (no `requirements.txt`, no
  `environment.yml`, no `pyproject.toml`). Imports across the three
  notebooks imply: `tensorflow`/`keras`, `pandas`, `numpy`, `scikit-learn`,
  `joblib`, `seaborn`, `matplotlib`, `hiplot`.
- **All paths are hardcoded absolute Windows paths** specific to the
  original author (`O:\VIT\...`, `C:\Users\varun\...`) — nothing is
  parameterized or portable.
- **CICFlowMeter is entirely external/manual** — no automation, no bundled
  jar, no documented version.
- **`packetsniff.sh` is Linux-only** and requires two explicit IP filters;
  it has no Windows equivalent in the repo.
- **Correctness bug in the prediction notebook:** the ANN was trained on
  `MinMaxScaler`-normalized features (`minmax.bin` is saved specifically for
  this), but `NIDS (Prediction).ipynb` never loads or applies that scaler —
  it casts the raw, unscaled CICFlowMeter values straight to `float32` and
  calls `model.predict()` directly. Consistent with this, the notebook's own
  saved output shows the **same 4 softmax values repeated for every single
  row** (`[0.471, 0.0021, 0.526, 0.0003]` for all 59,669 flows), i.e.
  inference has effectively collapsed to a constant — the pipeline as
  committed does not produce a working prediction. Fixing this (loading
  `minmax.bin` via `joblib.load` and calling `scaler.transform(df)` before
  `model.predict`) is required before this counts as "working reliably,"
  and is a straight bug-fix to existing logic, not a new model.

## Actual current execution flow (manual, cross-machine, today)

```
tcpdump (via packetsniff.sh, Linux, sudo)
  → .pcap
    → [MANUAL, external] CICFlowMeter GUI/CLI (not in repo)
      → .csv  (path hand-copied)
        → [MANUAL] edit hardcoded paths in NIDS (Prediction).ipynb
          → run notebook cells top to bottom
            → per-flow class prediction printed in-notebook
              (currently broken: unscaled input → constant output)
```

No component of this is wired to any other — every arrow above is a manual,
undocumented human step today.
