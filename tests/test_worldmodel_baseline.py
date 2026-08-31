import numpy as np
from backend.worldmodel.baseline import train_logistic_baseline, persistence_baseline, evaluate_baseline

def test_logistic_baseline_flattens_sequences_and_reports_metrics():
    rng=np.random.default_rng(42); X=rng.normal(size=(40,5,28)).astype("float32")
    y=np.repeat(np.arange(4),10)
    clf=train_logistic_baseline(X,y)
    report=evaluate_baseline(clf,X,y)
    assert clf.coef_.shape[1] == 5*28
    assert set(("macro_f1","precision","recall","false_positive_rate","per_class_f1")) <= set(report)

def test_persistence_uses_labels_not_scaled_features():
    meta={"input_labels":np.array([["BENIGN","DoS"],["DDoS","PortScan"]])}
    assert persistence_baseline(meta).tolist()==["DoS","PortScan"]
