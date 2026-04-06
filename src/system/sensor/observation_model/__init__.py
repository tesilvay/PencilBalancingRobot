from .base       import RegressionModel
from .null       import NullRegression
from .simple_dvs import (
    SimpleDVSRegressionModelLoader,
    SimpleDVSRegressionModelParams,
    SimpleRegressionParams,
    SIMPLE_REG_PRESETS,
)
from .simple_dvs_regression_model import SimpleDVSRegressionModel
from src.shared  import Spec, NullParams, NULL_PRESETS

REG_MODEL_REGISTRY = {
    "null":   Spec(NullRegression,                NullParams,                      NULL_PRESETS),
    "simple": Spec(SimpleDVSRegressionModelLoader, SimpleDVSRegressionModelParams, SIMPLE_REG_PRESETS),
}
