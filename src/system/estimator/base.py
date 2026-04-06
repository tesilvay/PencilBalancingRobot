from __future__ import annotations

import numpy as np

from src.shared import State, Measurement, ControlInput


class BaseEstimator:

    def estimate(
        self,
        y: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:
        raise NotImplementedError

    def reset(self, x_hat: State | None = None):
        raise NotImplementedError

    @staticmethod
    def measurement_z(y_meas: Measurement) -> np.ndarray:
        return np.asarray(y_meas.as_vector(), dtype=float).reshape(-1, 1)

    @staticmethod
    def control_u(u_cmd: ControlInput | None, y_meas: Measurement) -> np.ndarray:
        if u_cmd is None:
            return np.array([[y_meas.px], [y_meas.py]], dtype=float)
        return np.array([[u_cmd.px_cmd], [u_cmd.py_cmd]], dtype=float)

    def _step_prediction(
        self,
        z: np.ndarray,
        x_prev_col: np.ndarray,
        u_col: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-step state prediction and measurement innovation (same as Kalman pre-update)."""
        x_pred = self.A @ x_prev_col + self.B @ u_col
        innovation = (z - self.H @ x_pred).ravel()
        return innovation, x_pred
