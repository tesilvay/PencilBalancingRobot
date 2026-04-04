from src.shared import CameraPair, Measurement


class RegressionModel:
    """Base class for observation/regression models."""

    def estimate(self, cams: CameraPair) -> Measurement:
        raise NotImplementedError
