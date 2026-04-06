"""
Estimator package: concrete filters + ESTIMATOR_REGISTRY.

Each module defines its own ``*Params`` dataclass and ``*_PRESETS`` dict.
Plant and timing parameters use default_factory defaults; no plant registry
nesting on these Spec entries.
"""

from src.shared import Spec, NULL_PRESETS

from .base import BaseEstimator
from .fde import FDE_PRESETS, FdeParams, FiniteDifferenceEstimator

from .kalman import KALMAN_PRESETS, KalmanEstimator, KalmanParams
from .lpf import LPFParams, LPF_PRESETS, LowPassFiniteDifferenceEstimator

ESTIMATOR_REGISTRY = {
    "fde": Spec(FiniteDifferenceEstimator, FdeParams, FDE_PRESETS),
    "lpf": Spec(LowPassFiniteDifferenceEstimator, LPFParams, LPF_PRESETS),
    "kalman": Spec(KalmanEstimator, KalmanParams, KALMAN_PRESETS),
}

__all__ = [
    "BaseEstimator",
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
