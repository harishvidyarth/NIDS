"""ANN classifier metrics (backend/prediction/metrics.py)."""
from __future__ import annotations

import numpy as np

from backend.prediction.metrics import (
    _canon_label,
    compute_metrics,
    proxy_agreement_metrics,
)


def test_canon_label_collapses_cicids_sublabels():
    assert _canon_label("BENIGN") == "BENIGN"
    assert _canon_label("DDoS") == "DDoS"
    assert _canon_label("DoS Hulk") == "DoS"
    assert _canon_label("DoS GoldenEye") == "DoS"
    assert _canon_label("Heartbleed") == "DoS"
    assert _canon_label("PortScan") == "PortScan"
    # out of the ANN's 4-class scope -> dropped
    assert _canon_label("FTP-Patator") is None
    assert _canon_label("Web Attack \x96 Brute Force") is None
    assert _canon_label("Infiltration") is None


def test_compute_metrics_perfect_prediction():
    y = np.array(["BENIGN"] * 6 + ["DDoS"] * 3 + ["DoS"] * 1)
    m = compute_metrics(y, y.copy())
    assert m["accuracy"] == 1.0
    assert m["per_class"]["BENIGN"]["support"] == 6
    assert m["per_class"]["PortScan"]["support"] == 0        # absent class
    assert m["attack"]["false_positive_rate"] == 0.0
    assert m["confusion_matrix"][0][0] == 6


def test_compute_metrics_false_positive_rate():
    #                       one benign misread as DDoS -> FPR = 1 / 4 benign
    y_true = np.array(["BENIGN", "BENIGN", "BENIGN", "BENIGN", "DDoS"])
    y_pred = np.array(["BENIGN", "BENIGN", "BENIGN", "DDoS", "DDoS"])
    m = compute_metrics(y_true, y_pred)
    assert m["attack"]["false_positive_rate"] == 0.25
    assert m["attack"]["recall"] == 1.0


def test_proxy_agreement_metrics_flags_not_ground_truth():
    flows = [
        {"predicted_state": "BENIGN", "effective_state": "BENIGN"},
        {"predicted_state": "BENIGN", "effective_state": "DDoS"},   # signature escalated
        {"predicted_state": "DoS", "effective_state": "DoS"},
    ]
    m = proxy_agreement_metrics(flows)
    assert m["is_ground_truth"] is False
    assert m["source"] == "proxy_ann_vs_signature"
    assert 0.0 <= m["agreement_rate"] <= 1.0
    assert "PROXY" in m["note"]
