from backend.api import main


def test_multistep_route_returns_six_horizons(monkeypatch):
    payload = {
        "current_state": "BENIGN", "history_window_count": 5, "forecast_horizon": 6,
        "window_duration": None, "timestamp_mode": "row_order_proxy", "horizon_labeling": "windows",
        "horizons": [{"horizon": index, "seconds_ahead": None, "mitre_candidates": [],
                      "forecast_probability": 0.9, "mapping_confidence": None} for index in range(1, 7)],
        "early_warning_threshold": 0.8, "earliest_predicted_attack_horizon": None,
        "maximum_attack_probability": 0.1,
    }
    monkeypatch.setattr(main, "forecast_multistep_latest", lambda: payload)
    response = main.lstm_forecast_multistep()
    assert len(response["horizons"]) == 6
    assert all(item["seconds_ahead"] is None for item in response["horizons"])
    assert all(item["forecast_probability"] != item["mapping_confidence"] for item in response["horizons"])
