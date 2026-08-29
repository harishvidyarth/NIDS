# Phase 3 Dataset Download

The four CICIDS2017 CSV files used to train and evaluate the published Phase 3
LSTM are available as a GitHub Release asset because three raw files exceed
GitHub's 100 MB Git-object limit.

- Release: `phase3-datasets`
- Archive: `nids-cicids2017-phase3-source-csvs.tar.gz`
- Download: https://github.com/harishvidyarth/NIDS/releases/download/phase3-datasets/nids-cicids2017-phase3-source-csvs.tar.gz
- Archive SHA-256: `e3d10b574aab5fbb698f7d19b6e75a1f9208088905df609375c48ab2cfd65806`
- Upstream dataset: https://www.unb.ca/cic/datasets/ids-2017.html

## Setup

```bash
mkdir -p data/cicids2017
tar -xzf nids-cicids2017-phase3-source-csvs.tar.gz -C data/cicids2017
export NIDS_CICIDS2017_DIR="$PWD/data/cicids2017"
```

Verify the extracted files with `shasum -a 256 -c datasets/SHA256SUMS` from
the repository root after copying or extracting the four CSV files into
`data/cicids2017/`.

The archive includes only the public CICIDS2017 source CSVs used by Phase 3.
Local packet captures, uploads, runtime logs, virtual environments, and personal
traffic-derived files are not included.
