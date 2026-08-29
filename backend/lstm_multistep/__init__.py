"""Direct H1-H6 forecasting, isolated from the Phase 3 one-step model."""

from .training import forecast_latest, train_multistep

__all__ = ["forecast_latest", "train_multistep"]
