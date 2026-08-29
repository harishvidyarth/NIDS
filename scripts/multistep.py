from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.lstm_multistep.config import REPORT_ROOT
from backend.lstm_multistep.dataset import prepare_multistep_dataset
from backend.lstm_multistep.training import train_multistep


def main():
    parser = argparse.ArgumentParser(description="Phase 4 direct H1-H6 forecasting pipeline")
    parser.add_argument("command", choices=("dataset", "train", "evaluate", "benchmark", "all"))
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    if args.command == "dataset":
        _, manifest, _ = prepare_multistep_dataset(args.force_rebuild)
        print(json.dumps({"splits": manifest["splits"], "manifest": str(REPORT_ROOT.parent / "data/lstm_cache/multistep_dataset_manifest.json")}, indent=2))
        return
    if args.command in ("train", "all"):
        result = train_multistep(args.force_rebuild)
        print(json.dumps({"model_version": result["evaluation"]["model_version"], "evaluation_status": result["evaluation"]["evaluation_status"]}, indent=2))
        return
    report = REPORT_ROOT / ("multistep_evaluation_report.json" if args.command == "evaluate" else "multistep_performance_benchmark.json")
    if not report.is_file():
        raise SystemExit(f"Report not found; run train first: {report}")
    print(report.read_text())


if __name__ == "__main__": main()
