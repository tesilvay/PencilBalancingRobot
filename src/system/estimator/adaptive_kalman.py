from dataclasses import dataclass, field

import numpy as np

from src.shared import (
    ControlInput,
    Measurement,
    PlantParams,
    State,
    TimingParams,
    default_plant,
    default_timing,
)

from .base import BaseEstimator
from .dynamics_disc import discretize_AB, measurement_H


@dataclass
class AdaptiveKalmanParams:
    q_y_meas_pos: float
    q_y_meas_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_y_meas_pos: float
    r_y_meas_ang: float
    nis_alpha: float = 0.98
    nis_threshold: float = 13.3
    q_inflate_max: float = 50.0
    q_inflate_power: float = 1.0
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


ADAPTIVE_KALMAN_PRESETS = {
    "default": {
        "q_y_meas_pos": 1e-8,
        "q_y_meas_ang": 1e-8,
        "q_vel_pos": 1e-7,
        "q_vel_ang": 1e-7,
        "r_y_meas_pos": 1e-5,
        "r_y_meas_ang": 1e-5,
        "nis_alpha": 0.95,
        "nis_threshold": 3.0,
        "q_inflate_max": 100.0,
        "q_inflate_power": 1.25,
    },
    "sensitive": {
        "base": "default",
        "nis_threshold": 9.5,
        "nis_alpha": 0.95,
    },
    "aggressive": {
        "base": "default",
        "nis_threshold": 9.5,
        "q_inflate_max": 100.0,
        "q_inflate_power": 1.25,
    },
}


class AdaptiveKalmanEstimator(BaseEstimator):
    def __init__(self, params: AdaptiveKalmanParams):
        super().__init__()
        self._plant = params.plant
        self._disc_dt = float(params.timing.dt)
        self.A, self.B = discretize_AB(self._plant, self._disc_dt, mode="free")
        self.H = measurement_H()

        self.Q_base = np.diag([
            params.q_y_meas_pos, params.q_vel_pos, params.q_y_meas_ang, params.q_vel_ang,
            params.q_y_meas_pos, params.q_vel_pos, params.q_y_meas_ang, params.q_vel_ang,
        ])
        self.R = np.diag([
            params.r_y_meas_pos, params.r_y_meas_ang,
            params.r_y_meas_pos, params.r_y_meas_ang,
        ])

        self._nis_alpha = float(np.clip(params.nis_alpha, 0.0, 1.0))
        self._nis_threshold = float(max(params.nis_threshold, 1e-9))
        self._q_inflate_max = float(max(params.q_inflate_max, 1.0))
        self._q_inflate_power = float(max(params.q_inflate_power, 0.0))

        self.P_init = np.eye(8) * 2e-1
        self.x_hat_init_0 = np.zeros((8, 1))

        self.P = self.P_init.copy()
        self.x_hat = self.x_hat_init_0.copy()
        self._nis = 0.0
        self._nis_ewma = 0.0
        self._disturbance_score = 0.0
        self._q_scale = 1.0
        self._innovation_cov_ewma = self.R.copy()

    @property
    def nis(self) -> float:
        return self._nis

    @property
    def nis_ewma(self) -> float:
        return self._nis_ewma

    @property
    def q_scale(self) -> float:
        return self._q_scale

    @property
    def adaptive_lpf_weight(self) -> float:
        if self._q_inflate_max <= 1.0:
            return float(self.is_disturbed)
        span = self._q_inflate_max - 1.0
        return float(np.clip((self._q_scale - 1.0) / span, 0.0, 1.0))

    @property
    def is_disturbed(self) -> bool:
        return self._disturbance_score > self._nis_threshold

    @staticmethod
    def _safe_invert(S: np.ndarray) -> np.ndarray:
        S = 0.5 * (S + S.T)
        try:
            return np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S)

    def _safe_solve(self, S: np.ndarray, y: np.ndarray) -> np.ndarray:
        S = 0.5 * (S + S.T)
        try:
            return np.linalg.solve(S, y)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S) @ y

    def _ensure_discretization(self, dt: float) -> None:
        dt = float(dt)
        if np.isclose(dt, self._disc_dt, rtol=0.0, atol=1e-12):
            return
        self.A, self.B = discretize_AB(self._plant, dt, mode="free")
        self._disc_dt = dt

    def _update_disturbance_metrics(self, innovation: np.ndarray, S_nominal: np.ndarray) -> None:
        innovation = np.asarray(innovation, dtype=float).reshape(-1, 1)
        nis_vec = self._safe_solve(S_nominal, innovation)
        self._nis = float((innovation.T @ nis_vec).item())
        self._nis_ewma = (
            self._nis_alpha * self._nis_ewma
            + (1.0 - self._nis_alpha) * self._nis
        )
        self._disturbance_score = max(self._nis, self._nis_ewma)
        self._innovation_cov_ewma = (
            self._nis_alpha * self._innovation_cov_ewma
            + (1.0 - self._nis_alpha) * (innovation @ innovation.T)
        )

    def _compute_q_scale(self) -> float:
        if self._disturbance_score <= self._nis_threshold:
            return 1.0

        ratio = self._disturbance_score / self._nis_threshold
        scale = ratio ** self._q_inflate_power if self._q_inflate_power > 0.0 else ratio
        return float(np.clip(scale, 1.0, self._q_inflate_max))

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:
        self._ensure_discretization(dt)

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)

        x_pred = self.A @ self.x_hat + self.B @ u
        innovation = z - self.H @ x_pred

        P_pred_nominal = self.A @ self.P @ self.A.T + self.Q_base
        S_nominal = self.H @ P_pred_nominal @ self.H.T + self.R
        self._update_disturbance_metrics(innovation, S_nominal)

        self._q_scale = self._compute_q_scale()
        Q_eff = self.Q_base * self._q_scale
        P_pred = self.A @ self.P @ self.A.T + Q_eff

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ self._safe_invert(S)

        self.x_hat = x_pred + K @ innovation
        eye = np.eye(self.P.shape[0])
        innovation_projector = eye - K @ self.H
        self.P = (
            innovation_projector @ P_pred @ innovation_projector.T
            + K @ self.R @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)

        return State.from_iterable(self.x_hat.flatten()), innovation.ravel()

    def reset(self, x_hat: State | None = None):
        if x_hat is not None:
            self.x_hat = x_hat.as_vector().reshape(-1, 1)
        else:
            self.x_hat = self.x_hat_init_0.copy()

        self.P = self.P_init.copy()
        self._nis = 0.0
        self._nis_ewma = 0.0
        self._disturbance_score = 0.0
        self._q_scale = 1.0
        self._innovation_cov_ewma = self.R.copy()
