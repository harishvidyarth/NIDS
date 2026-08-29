from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.lstm.rigorous_evaluation import evaluate_saved_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved next-state LSTM without training it.")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = evaluate_saved_artifact(args.artifact_dir, args.json, args.markdown)
    summary = {
        "model_version": report["model_identity"]["model_version"],
        "leakage_audit": report["leakage_audit"]["overall"],
        "interpretation_status": report["interpretation_status"],
        "test_distribution": report["evaluations"]["test"]["class_distribution"],
        "test_lstm": {
            key: report["evaluations"]["test"]["lstm"][key]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "attack_containing_evaluation": report["attack_containing_evaluation"]["status"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
