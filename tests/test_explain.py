"""Gradient x input feature attribution for the ANN."""
from __future__ import annotations

import numpy as np
import pytest

from backend.prediction.explain import attribute, driving_features, top_features_for_row
from backend.prediction.features import TRAINING_FEATURES
from backend.prediction.predict import _load_artifacts

N_FEATURES = len(TRAINING_FEATURES)


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("tensorflow")
    mdl, _ = _load_artifacts()
    return mdl


def test_attribution_shape_and_finiteness(model):
    rng = np.random.default_rng(0)
    scaled = rng.random((5, N_FEATURES)).astype(np.float32)
    probs = model.predict(scaled, verbose=0)
    idx = np.argmax(probs, axis=1)
    attr = attribute(model, scaled, idx)
    assert attr.shape == (5, N_FEATURES)
    assert np.isfinite(attr).all()


def test_attribution_is_deterministic(model):
    rng = np.random.default_rng(1)
    scaled = rng.random((3, N_FEATURES)).astype(np.float32)
    idx = np.argmax(model.predict(scaled, verbose=0), axis=1)
    a = attribute(model, scaled, idx)
    b = attribute(model, scaled, idx)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_top_features_ranked_and_named(model):
    rng = np.random.default_rng(2)
    scaled = rng.random((1, N_FEATURES)).astype(np.float32)
    idx = np.argmax(model.predict(scaled, verbose=0), axis=1)
    attr = attribute(model, scaled, idx)[0]
    top = top_features_for_row(attr, k=8)
    assert len(top) == 8
    known = {name.strip() for name in TRAINING_FEATURES}
    assert all(item["feature"] in known for item in top)
    mags = [abs(item["contribution"]) for item in top]
    assert mags == sorted(mags, reverse=True)


def test_driving_features_aggregate(model):
    rng = np.random.default_rng(3)
    scaled = rng.random((12, N_FEATURES)).astype(np.float32)
    idx = np.argmax(model.predict(scaled, verbose=0), axis=1)
    attr = attribute(model, scaled, idx)
    drv = driving_features(attr, k=10)
    assert len(drv) == 10
    vals = [item["mean_abs_contribution"] for item in drv]
    assert vals == sorted(vals, reverse=True)
    assert all(v >= 0 for v in vals)


def test_driving_features_empty_input():
    assert driving_features(np.empty((0, N_FEATURES))) == []
