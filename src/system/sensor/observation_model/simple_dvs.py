from dataclasses import dataclass

from src.shared import CameraPair, PoseMeasurement

from .base import RegressionModel
from .simple_dvs_regression_model import SimpleDVSRegressionModel


@dataclass
class SimpleDVSRegressionModelParams:
    calibration_path: str


SimpleRegressionParams = SimpleDVSRegressionModelParams

SIMPLE_REG_PRESETS = {
    "default": {
        "calibration_path": "hardware/calibration_files/dvs_affine_calibration.json",
    }
}


class SimpleDVSRegressionModelLoader(RegressionModel):
    """Registry-facing wrapper: loads a frozen SimpleDVSRegressionModel from disk."""

    def __init__(self, params: SimpleDVSRegressionModelParams):
        self._model = SimpleDVSRegressionModel.load(params.calibration_path)

    def estimate(self, cams: CameraPair) -> PoseMeasurement:
        return self._model.estimate(cams)
