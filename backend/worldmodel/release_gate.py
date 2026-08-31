"""Fail-closed admission checks for a versioned world-model artifact."""
from __future__ import annotations
import hashlib
import numpy as np

MIN_SUPPORT = 30  # avoids claims from single-digit holdout classes
MAX_TRAIN_IMBALANCE = 10.0  # recorded warning; data diagnostic only
MIN_H1_MACRO_F1 = .60  # minimum useful one-step discrimination
MIN_BALANCED_ACCURACY = .55  # protects minority classes
MAX_BENIGN_FPR = .10  # avoids an unusable alert stream
MIN_CLASS_RECALL = .40  # every advertised class must be detectable
MIN_HORIZON_F1 = .40  # rollout must retain forecasting value
MAX_HORIZON_DEGRADATION = .25  # prevent sharp autoregressive collapse
MAX_NORMALIZED_STATE_MAE = .90  # bounded scaled-state forecast error
MAX_ACTIVE_REGRESSION = .02  # no material regression from active artifact

def _seq_hashes(split):
    vals = split.get("sequence_hashes", []) if isinstance(split, dict) else []
    return set(vals)

def _support(split, classes):
    if isinstance(split, dict): return split.get("class_support", {})
    return {}

def run_release_gate(metrics, train_split, valid_split, test_split, classes, active_metrics=None):
    failures=[]; checks={}
    def check(name, passed, **extra):
        checks[name]={"passed": bool(passed), **extra}
        if not passed: failures.append(name.replace("_", " "))
    hashes=[_seq_hashes(x) for x in (train_split, valid_split, test_split)]
    check("sequence_disjoint", not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2]))
    for name, split in (("validation_support",valid_split),("test_support",test_split)):
        support=_support(split,classes); check(name, all(int(support.get(c,0)) >= MIN_SUPPORT for c in classes), support=support, minimum=MIN_SUPPORT)
    leakage = metrics.get("leakage_audit", {})
    check("leakage_audit", leakage.get("passed", True), detail=leakage)
    train_support=_support(train_split,classes); positive=[int(train_support.get(c,0)) for c in classes if int(train_support.get(c,0))]
    ratio=max(positive)/min(positive) if positive else float("inf")
    checks["train_imbalance"]={"passed": True,"warning":ratio > MAX_TRAIN_IMBALANCE,"ratio":ratio}
    one=metrics.get("one_step", metrics)
    check("one_step_quality", one.get("macro_f1",0)>=MIN_H1_MACRO_F1 and one.get("balanced_accuracy",0)>=MIN_BALANCED_ACCURACY)
    check("benign_false_positive_rate", one.get("false_positive_rate",1)<=MAX_BENIGN_FPR)
    recalls=one.get("per_class_recall", {c: one.get("per_class",{}).get(c,{}).get("recall",0) for c in classes})
    check("per_class_recall", all(recalls.get(c,0)>=MIN_CLASS_RECALL for c in classes), recalls=recalls)
    horizon=metrics.get("horizon_k", {})
    check("horizon_quality", horizon.get("macro_f1",0)>=MIN_HORIZON_F1 and horizon.get("macro_f1",0)>=one.get("macro_f1",0)-MAX_HORIZON_DEGRADATION and horizon.get("normalized_state_mae",float("inf"))<=MAX_NORMALIZED_STATE_MAE)
    benchmark=metrics.get("benchmark",{})
    check("beats_logistic", benchmark.get("world_model",{}).get("macro_f1",0)>benchmark.get("logistic_regression",{}).get("macro_f1",float("inf")))
    check("save_load_parity", bool(metrics.get("save_load_parity", False)))
    if active_metrics:
        check("active_regression", horizon.get("macro_f1",0)>=active_metrics.get("horizon_k",{}).get("macro_f1",horizon.get("macro_f1",0))-MAX_ACTIVE_REGRESSION)
    return {"passed": not failures, "failures": failures, "checks": checks}
