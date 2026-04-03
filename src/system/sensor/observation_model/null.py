from src.shared import CameraPair, PoseMeasurement

from .base import RegressionModel


class NullRegression(RegressionModel):
    def __init__(self, params=None):
        pass

    def estimate(self, cams: CameraPair) -> PoseMeasurement:
        return None
