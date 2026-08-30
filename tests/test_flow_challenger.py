from __future__ import annotations

import numpy as np

from backend.prediction.flow_challenger import fit_challenger


def test_hist_gradient_boosting_challenger_reports_activation_metrics():
    rng = np.random.default_rng(42)
    values, labels = [], []
    for index, label in enumerate(("BENIGN", "DDoS", "DoS", "PortScan")):
        values.append(rng.normal(loc=index * 3, scale=0.5, size=(40, 6)))
        labels.extend([label] * 40)
    model, metrics, background = fit_challenger(np.concatenate(values), np.asarray(labels))
    assert set(model.classes_) == {"BENIGN", "DDoS", "DoS", "PortScan"}
    assert {"macro_f1", "ddos_recall", "attack_pr_auc", "benign_false_positive_rate"}.issubset(metrics)
    assert background.shape == (100, 6)
