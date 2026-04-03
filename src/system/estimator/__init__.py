"""
Estimator package: concrete filters + ESTIMATOR_REGISTRY.

Each module defines its own ``*Params`` dataclass and ``*_PRESETS`` dict.
Plant and timing parameters use default_factory defaults; no plant registry
nesting on these Spec entries.
"""

from src.shared import Spec, NullParams, NULL_PRESETS

from .base       import BaseEstimator
from .fde        import FDEParams,        FDE_PRESETS,        FiniteDifferenceEstimator
from .full_kalman import (
    FULL_KALMAN_PRESETS,
    FullKalmanParams,
    FullStateKalmanFilter,
)
from .kalman     import KALMAN_PRESETS,    KalmanEstimator,   KalmanParams
from .kalman_core import KalmanStepResult, run_linear_kalman_step
from .lpf        import LPFParams,         LPF_PRESETS,       LowPassFiniteDifferenceEstimator

ESTIMATOR_REGISTRY = {
    "fde":         Spec(FiniteDifferenceEstimator,        FDEParams,        FDE_PRESETS),
    "lpf":         Spec(LowPassFiniteDifferenceEstimator, LPFParams,        LPF_PRESETS),
    "kalman":      Spec(KalmanEstimator,                  KalmanParams,     KALMAN_PRESETS),
    "full_kalman": Spec(FullStateKalmanFilter,            FullKalmanParams, FULL_KALMAN_PRESETS),
}

__all__ = [
    "BaseEstimator",
    "FiniteDifferenceEstimator",
    "FDEParams",
    "FDE_PRESETS",
    "LowPassFiniteDifferenceEstimator",
    "LPFParams",
    "LPF_PRESETS",
    "KalmanEstimator",
    "KalmanParams",
    "KALMAN_PRESETS",
    "FullStateKalmanFilter",
    "FullKalmanParams",
    "FULL_KALMAN_PRESETS",
    "KalmanStepResult",
    "run_linear_kalman_step",
    "ESTIMATOR_REGISTRY",
]
