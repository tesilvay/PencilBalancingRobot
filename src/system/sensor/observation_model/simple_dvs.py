from dataclasses import dataclass

from src.shared import CameraPair, Measurement

from .base import RegressionModel
from .simple_dvs_regression_model import SimpleDVSRegressionModel


@dataclass
class SimpleDVSRegressionModelParams:
    calibration_path: str
    max_tilt_deg: float = 30.0


SimpleRegressionParams = SimpleDVSRegressionModelParams

SIMPLE_REG_PRESETS = {
    "default": {
        "calibration_path": "src/system/sensor/observation_model/calibration_files/simple_dvs_regression.json",
        "max_tilt_deg": 30.0,
    }
}


class SimpleDVSRegressionModelLoader(RegressionModel):
    """Registry-facing wrapper: loads a frozen SimpleDVSRegressionModel from disk."""

    def __init__(self, params: SimpleDVSRegressionModelParams):
        self._model = SimpleDVSRegressionModel.load(
            params.calibration_path,
            max_tilt_deg=params.max_tilt_deg,
        )

    def estimate(self, cams: CameraPair) -> Measurement:
        return self._model.estimate(cams)
