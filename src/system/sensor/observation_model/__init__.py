from .base       import RegressionModel
from .null       import NullRegression
from .simple_dvs import SimpleDVSRegressionModel, SimpleRegressionParams, SIMPLE_REG_PRESETS
from src.shared  import Spec, NullParams, NULL_PRESETS

REG_MODEL_REGISTRY = {
    "none":   Spec(NullRegression,          NullParams,   NULL_PRESETS),
    "simple": Spec(SimpleDVSRegressionModel, SimpleRegressionParams, SIMPLE_REG_PRESETS),
}
