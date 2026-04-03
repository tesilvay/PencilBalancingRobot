from src.shared import CameraPair, PoseMeasurement


class RegressionModel:
    """Base class for observation/regression models."""

    def estimate(self, cams: CameraPair) -> PoseMeasurement:
        raise NotImplementedError
