from src.shared import CameraPair, NullParams, PoseMeasurement

from .base import RegressionModel


class NullRegression(RegressionModel):
    def __init__(self, params: NullParams):
        pass

    def estimate(self, cams: CameraPair) -> PoseMeasurement:
        return None
