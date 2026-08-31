from pathlib import Path

from backend.temporal.schema import STATE_FEATURE_NAMES, STATE_FEATURE_NAMES_V2
from backend.temporal.temporal_dataset import prepare_temporal_dataset
from tests.test_temporal import make_flow_csv


def test_v2_feature_order_is_additive():
    assert STATE_FEATURE_NAMES_V2[:28] == STATE_FEATURE_NAMES
    assert STATE_FEATURE_NAMES_V2[28:] == [
        "dns_entropy", "unique_sni", "beacon_score", "byte_asymmetry",
        "ja3_novelty", "http_error_rate",
    ]


def test_prepare_enrichment_on_and_off(tmp_path: Path):
    csv_path = tmp_path / "session.csv"
    make_flow_csv(100).to_csv(csv_path, index=False)
    enrichment = {
        "dns_query_entropy_mean": 3.2,
        "unique_sni_count": 7,
        "beacon_score_max": 0.91,
        "byte_asymmetry_max": 0.8,
        "ja3_novelty": 0.5,
    }
    legacy = prepare_temporal_dataset(csv_path, tmp_path / "legacy", sequence_length=5,
                                      enrich_windows=False)
    enriched = prepare_temporal_dataset(csv_path, tmp_path / "v2", sequence_length=5,
                                        enrich_windows=True, ingest_enrichment=enrichment)
    assert legacy["state_features"] == 28
    assert "dns_entropy" not in legacy["states_df"]
    assert enriched["state_features"] == 34
    assert enriched["forecast_state_features"] == 28
    assert set(STATE_FEATURE_NAMES_V2).issubset(enriched["states_df"].columns)
    assert enriched["states_df"]["beacon_score"].eq(0.91).all()
    assert enriched["states_df"]["http_error_rate"].eq(0.0).all()
    assert enriched["sequences"]["X"].shape[-1] == 28
