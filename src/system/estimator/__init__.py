"""
Estimator package: concrete filters + ESTIMATOR_REGISTRY.

Each module defines its own ``*Params`` dataclass and ``*_PRESETS`` dict.
Plant and timing parameters use default_factory defaults; no plant registry
nesting on these Spec entries.
"""

from src.shared import Spec, NULL_PRESETS

from .base import BaseEstimator
from .adaptive_kalman import (
    ADAPTIVE_KALMAN_PRESETS,
    AdaptiveKalmanEstimator,
    AdaptiveKalmanParams,
)
from .fde import FDE_PRESETS, FdeParams, FiniteDifferenceEstimator

from .kalman import KALMAN_PRESETS, KalmanEstimator, KalmanParams
from .lpf import LPFParams, LPF_PRESETS, LowPassFiniteDifferenceEstimator
from .augmented_kalman import AugKalmanParams, AUG_KALMAN_PRESETS, AugmentedKalmanEstimator

ESTIMATOR_REGISTRY = {
    "fde": Spec(FiniteDifferenceEstimator, FdeParams, FDE_PRESETS),
    "lpf": Spec(LowPassFiniteDifferenceEstimator, LPFParams, LPF_PRESETS),
    "kalman": Spec(KalmanEstimator, KalmanParams, KALMAN_PRESETS),
    "adaptive_kalman": Spec(
        AdaptiveKalmanEstimator,
        AdaptiveKalmanParams,
        ADAPTIVE_KALMAN_PRESETS,
    ),
    "augmented_kalman": Spec(
        AugmentedKalmanEstimator,
        AugKalmanParams,
        AUG_KALMAN_PRESETS,
        
    ),
}

__all__ = [
    "BaseEstimator",
    "AdaptiveKalmanEstimator",
    "AdaptiveKalmanParams",
    "ADAPTIVE_KALMAN_PRESETS",
    "AugmentedKalmanEstimator",
    "AugKalmanParams",
    "AUG_KALMAN_PRESETS",
    "FiniteDifferenceEstimator",
    "FdeParams",
    "FDE_PRESETS",
    "LowPassFiniteDifferenceEstimator",
    "LPFParams",
    "LPF_PRESETS",
    "KalmanEstimator",
    "KalmanParams",
    "KALMAN_PRESETS",
    "ESTIMATOR_REGISTRY",
]
