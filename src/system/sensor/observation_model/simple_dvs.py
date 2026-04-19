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
        m = self._model
        if m.pos_v3 is not None:
            v3 = m.pos_v3
            print(
                f"[ObsModel] simple_dvs_regression_v3 (TPS) loaded from {params.calibration_path}\n"
                f"  {len(v3.positions_m)} position pts, "
                f"{len(v3.tilt_positions_m)}x{len(v3.tilt_alphas_rad)} tilt grid"
            )
        elif m.pos_v2 is not None:
            print(
                f"[ObsModel] simple_dvs_regression_v2 (bilinear) loaded from {params.calibration_path}\n"
                f"  px: a0={m.pos_v2.a0:.6f}  a1(b1)={m.pos_v2.a1:.6f}  a2(b2)={m.pos_v2.a2:.6f}\n"
                f"  py: c0={m.pos_v2.c0:.6f}  c1(b2)={m.pos_v2.c1:.6f}  c2(b1)={m.pos_v2.c2:.6f}"
            )
        else:
            print(
                f"[ObsModel] simple_dvs_regression_v1 (affine) loaded from {params.calibration_path}\n"
                f"  cam1: k_pos={m.cam1.k_pos:.6f}  b_pos={m.cam1.b_pos:.6f}\n"  # type: ignore[union-attr]
                f"  cam2: k_pos={m.cam2.k_pos:.6f}  b_pos={m.cam2.b_pos:.6f}"  # type: ignore[union-attr]
            )

    def estimate(self, cams: CameraPair) -> Measurement:
        return self._model.estimate(cams)
