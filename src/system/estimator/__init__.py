"""
Estimator package: concrete filters + ESTIMATOR_REGISTRY (assembly).

Each module defines its own ``*Params`` dataclass and ``*_PRESETS`` dict.
"""

from src.shared import Spec
from src.system.plant import PLANT_REGISTRY

from .base import BaseEstimator
from .fde import FDEParams, FDE_PRESETS, FiniteDifferenceEstimator
from .full_kalman import (
    FULL_KALMAN_PRESETS,
    FullKalmanParams,
    FullStateKalmanFilter,
)
from .kalman import KALMAN_PRESETS, KalmanEstimator, KalmanParams
from .kalman_core import KalmanStepResult, run_linear_kalman_step
from .lpf import LPFParams, LPF_PRESETS, LowPassFiniteDifferenceEstimator

ESTIMATOR_REGISTRY = {
    "fde":         Spec(FiniteDifferenceEstimator,        FDEParams,        FDE_PRESETS,         registries={"plant": PLANT_REGISTRY}),
    "lpf":         Spec(LowPassFiniteDifferenceEstimator, LPFParams,        LPF_PRESETS,         registries={"plant": PLANT_REGISTRY}),
    "kalman":      Spec(KalmanEstimator,                  KalmanParams,     KALMAN_PRESETS,      registries={"plant": PLANT_REGISTRY}),
    "full_kalman": Spec(FullStateKalmanFilter,            FullKalmanParams, FULL_KALMAN_PRESETS,  registries={"plant": PLANT_REGISTRY}),
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
