from dataclasses import dataclass, field
from collections import deque

import numpy as np

from src.shared import (
    State,
    Measurement,
    ControlInput,
    PlantParams,
    TimingParams,
    default_plant,
    default_timing,
)

from .base import BaseEstimator
from .dynamics_disc import discretize_AB, measurement_H


@dataclass
class LPFParams:
    alpha_meas: float
    alpha_vel: float
    history_size: int | None = None
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


LPF_PRESETS = {
    "default": {"alpha_meas": 0.0, "alpha_vel": 0.2, "history_size": 8},
    "lead": {"alpha_meas": 0.0, "alpha_vel": 0.1},
    "fde": {"alpha_meas": 0.0, "alpha_vel": 0.1, "history_size": 2},
    "test": {"alpha_meas": 0.75, "alpha_vel": 0.75},
    "smoother": {"alpha_meas": 0.8, "alpha_vel": 0.8},
    "test2": {"alpha_meas": 0.0, "alpha_vel": 0.95},
}


class LowPassFiniteDifferenceEstimator(BaseEstimator):
    def __init__(self, params: LPFParams):
        super().__init__()
        self.alpha_meas = params.alpha_meas
        self.alpha_vel = params.alpha_vel
        self._plant = params.plant
        self._disc_dt = float(params.timing.dt)
        self.history_size = self._resolve_history_size(params.history_size, params.timing)
        self.A, self.B = discretize_AB(self._plant, self._disc_dt)
        self.H = measurement_H()

        self._x_post = np.zeros((8, 1))
        self._sample_time = 0.0
        self._y_history: deque[np.ndarray] = deque(maxlen=self.history_size)
        self._time_history: deque[float] = deque(maxlen=self.history_size)
        self.prev_y_filt: np.ndarray | None = None
        self.prev_vel = np.zeros(4)

    @staticmethod
    def _resolve_history_size(history_size: int | None, timing: TimingParams) -> int:
        if history_size is not None:
            history_size = int(history_size)
            if history_size <= 0:
                raise ValueError("LPF history_size must be positive when provided.")
            return history_size

        dt = float(timing.dt)
        actuator_dt = float(getattr(timing, "actuator_dt", dt))
        if dt <= 0.0:
            raise ValueError("LPF timing.dt must be positive.")
        ratio = actuator_dt / dt if actuator_dt > 0.0 else 1.0
        return max(2, int(round(ratio)))

    def _append_history(self, y_vec: np.ndarray, dt: float) -> None:
        if self._time_history:
            self._sample_time += float(dt)
        else:
            self._sample_time = 0.0

        self._y_history.append(np.asarray(y_vec, dtype=float).copy())
        self._time_history.append(self._sample_time)

    def _fit_history(self) -> tuple[np.ndarray, np.ndarray]:
        y_hist = np.vstack(self._y_history)
        if y_hist.shape[0] == 1:
            return y_hist[-1].copy(), np.zeros(4)

        tau = np.asarray(self._time_history, dtype=float)
        tau = tau - tau[-1]
        tau_centered = tau - tau.mean()
        denom = float(np.dot(tau_centered, tau_centered))
        if denom <= 1e-18:
            return y_hist[-1].copy(), np.zeros(4)

        y_mean = y_hist.mean(axis=0)
        slope = (tau_centered[:, None] * (y_hist - y_mean)).sum(axis=0) / denom
        current = y_mean - slope * tau.mean()
        return current, slope

    def _ensure_discretization(self, dt: float) -> None:
        dt = float(dt)
        if np.isclose(dt, self._disc_dt, rtol=0.0, atol=1e-12):
            return
        self.A, self.B = discretize_AB(self._plant, dt)
        self._disc_dt = dt

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:
        self._ensure_discretization(dt)

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)
        innovation, _ = self._step_prediction(z, self._x_post, u)

        y_vec = y_meas.as_vector()
        self._append_history(y_vec, dt)
        y_hist_filt, raw_vel = self._fit_history()

        if self.prev_y_filt is None:
            y_filt = y_hist_filt
            vel = raw_vel
        else:
            y_filt = self.alpha_meas * self.prev_y_filt + (1 - self.alpha_meas) * y_hist_filt
            vel = self.alpha_vel * self.prev_vel + (1 - self.alpha_vel) * raw_vel

        x_hat = State(
            px=y_filt[0], vx=vel[0],
            ax=y_filt[1], wx=vel[1],
            py=y_filt[2], vy=vel[2],
            ay=y_filt[3], wy=vel[3],
        )

        self.prev_y_filt = y_filt
        self.prev_vel = vel
        self._x_post = x_hat.as_vector().reshape(-1, 1)

        return x_hat, innovation

    def reset(self, x_hat: State | None = None):
        self._sample_time = 0.0
        self._y_history.clear()
        self._time_history.clear()

        if x_hat is None:
            self._x_post = np.zeros((8, 1))
            self.prev_y_filt = None
            self.prev_vel = np.zeros(4)
        else:
            self._x_post = x_hat.as_vector().reshape(-1, 1)
            self.prev_y_filt = np.array([x_hat.px, x_hat.ax, x_hat.py, x_hat.ay])
            self.prev_vel = np.array([x_hat.vx, x_hat.wx, x_hat.vy, x_hat.wy])
            self._y_history.append(self.prev_y_filt.copy())
            self._time_history.append(self._sample_time)
