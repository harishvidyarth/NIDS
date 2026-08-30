from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.prediction.shap_service import ExplanationJobs, stratified_background, tree_explanation


def test_stratified_background_is_deterministic_and_bounded():
    values = np.arange(800, dtype=float).reshape(200, 4)
    labels = np.array(["BENIGN"] * 150 + ["DDoS"] * 50)
    first = stratified_background(values, labels)
    second = stratified_background(values, labels)
    assert first.shape == (100, 4)
    np.testing.assert_array_equal(first, second)
    selected = {tuple(row) for row in first}
    assert any(tuple(row) in selected for row in values[150:])


def test_tree_shap_returns_signed_ranked_features():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(160, 4))
    labels = np.where(values[:, 0] + values[:, 1] > 0, "DDoS", "BENIGN")
    model = HistGradientBoostingClassifier(random_state=42).fit(values, labels)
    result = tree_explanation(model, values[:100], values[100:101], ["a", "b", "c", "d"], "DDoS")
    assert result["method"] == "shap.TreeExplainer"
    assert result["is_shap"] is True
    assert len(result["feature_contributions"]) == 4
    magnitudes = [abs(item["contribution"]) for item in result["feature_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_async_jobs_cache_by_model_and_input_hash():
    store = ExplanationJobs(workers=1)
    values = np.ones((1, 4))
    first = store.submit("model-a", values, {"class": "DDoS"}, lambda: {"ok": True})
    for _ in range(100):
        status = store.get(first["job_id"])
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["result"] == {"ok": True}
    cached = store.submit("model-a", values, {"class": "DDoS"}, lambda: {"ok": False})
    assert cached["job_id"] == first["job_id"]
    assert cached["cache_hit"] is True


def test_api_exposes_async_explanation_endpoints():
    source = open("backend/api/main.py").read()
    assert '@app.post("/api/explanations", status_code=202)' in source
    assert '@app.get("/api/explanations/{job_id}")' in source
