from src.shared import CameraPair, NullParams, Measurement

from .base import RegressionModel


class NullRegression(RegressionModel):
    def __init__(self, params: NullParams):
        pass

    def estimate(self, cams: CameraPair) -> Measurement:
        return None
