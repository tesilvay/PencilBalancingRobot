from dataclasses import dataclass

from src.system.sensor.observation_model.simple_dvs_regression_model import (
    SimpleDVSRegressionModel,
)


@dataclass
class SimpleRegressionParams:
    calibration_path: str


SIMPLE_REG_PRESETS = {
    "default": {
        "calibration_path": "hardware/calibration_files/dvs_affine_calibration.json",
    }
}
