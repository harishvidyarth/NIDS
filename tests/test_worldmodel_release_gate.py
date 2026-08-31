from backend.worldmodel.release_gate import run_release_gate

CLASSES=["BENIGN","DDoS","DoS","PortScan"]
def split(prefix): return {"class_support":{c:30 for c in CLASSES},"sequence_hashes":[prefix+str(i) for i in range(4)]}
def metrics():
    one={"macro_f1":.7,"balanced_accuracy":.7,"false_positive_rate":.05,"per_class_recall":{c:.7 for c in CLASSES}}
    return {"one_step":one,"horizon_k":{"macro_f1":.5,"normalized_state_mae":.2},"save_load_parity":True,"leakage_audit":{"passed":True},"benchmark":{"world_model":{"macro_f1":.7},"logistic_regression":{"macro_f1":.6}}}
def test_release_gate_passes_complete_candidate():
    assert run_release_gate(metrics(),split("t"),split("v"),split("e"),CLASSES)["passed"]
def test_release_gate_rejects_duplicate_and_low_recall():
    value=metrics(); value["one_step"]["per_class_recall"]["DoS"]=.2
    result=run_release_gate(value,split("x"),split("x"),split("e"),CLASSES)
    assert not result["passed"]
    assert not result["checks"]["sequence_disjoint"]["passed"]
    assert not result["checks"]["per_class_recall"]["passed"]
