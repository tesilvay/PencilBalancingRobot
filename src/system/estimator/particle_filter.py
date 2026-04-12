from dataclasses import dataclass, field

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
class ParticleFilterParams:
    num_particles: int
    q_y_meas_pos: float
    q_y_meas_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_y_meas_pos: float
    r_y_meas_ang: float
    init_pos_std: float = 5e-2
    init_vel_std: float = 2e-1
    init_ang_std: float = 1.0e-1
    init_ang_vel_std: float = 5e-1
    resample_threshold: float = 1.0
    random_seed: int | None = None
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


PARTICLE_FILTER_PRESETS = {
    "default": {
        "num_particles": 100,
        "q_y_meas_pos": 1e-6,
        "q_y_meas_ang": 1e-6,
        "q_vel_pos": 1e-2,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-2,
        "r_y_meas_ang": 1e-2,
    },
    "test": {
        "base": "default",
        "num_particles": 500,
        "q_y_meas_pos": 1e-4,
        "q_y_meas_ang": 1e-4,
        "q_vel_pos": 1e-2,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-3,
        "r_y_meas_ang": 1e-3,
        "random_seed": 1,
    },
    "fast": {
        "base": "default",
        "num_particles": 250,
    },
}


class ParticleFilterEstimator(BaseEstimator):
    """Bootstrap/SIR particle filter with Gaussian process and measurement noise."""

    def __init__(self, params: ParticleFilterParams):
        super().__init__()
        self._plant = params.plant
        self._disc_dt = float(params.timing.dt)
        self.A, self.B = discretize_AB(self._plant, self._disc_dt)
        self.H = measurement_H()

        self.num_particles = int(params.num_particles)
        if self.num_particles <= 0:
            raise ValueError("Particle filter num_particles must be positive.")

        self.Q = self._build_process_noise(params)
        self.R = self._build_measurement_noise(params)
        self._process_std = self._sqrt_diag(self.Q, "process noise")
        self._measurement_var = np.diag(self.R).astype(float)
        if np.any(self._measurement_var <= 0.0):
            raise ValueError("Particle filter measurement variances must be positive.")

        self._init_std = np.array([
            params.init_pos_std,
            params.init_vel_std,
            params.init_ang_std,
            params.init_ang_vel_std,
            params.init_pos_std,
            params.init_vel_std,
            params.init_ang_std,
            params.init_ang_vel_std,
        ], dtype=float)
        if np.any(self._init_std < 0.0):
            raise ValueError("Particle filter initial standard deviations must be nonnegative.")

        self._resample_threshold = float(params.resample_threshold)
        if not 0.0 <= self._resample_threshold <= 1.0:
            raise ValueError("Particle filter resample_threshold must be in [0, 1].")

        self._rng = np.random.default_rng(params.random_seed)
        self.particles = np.zeros((self.num_particles, 8), dtype=float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=float)
        self.x_hat = np.zeros((8, 1), dtype=float)
        self.reset()

    @staticmethod
    def _build_process_noise(params: ParticleFilterParams) -> np.ndarray:
        return np.diag([
            params.q_y_meas_pos, params.q_vel_pos, params.q_y_meas_ang, params.q_vel_ang,
            params.q_y_meas_pos, params.q_vel_pos, params.q_y_meas_ang, params.q_vel_ang,
        ])

    @staticmethod
    def _build_measurement_noise(params: ParticleFilterParams) -> np.ndarray:
        return np.diag([
            params.r_y_meas_pos, params.r_y_meas_ang,
            params.r_y_meas_pos, params.r_y_meas_ang,
        ])

    @staticmethod
    def _sqrt_diag(covariance: np.ndarray, name: str) -> np.ndarray:
        diag = np.diag(covariance).astype(float)
        if np.any(diag < 0.0):
            raise ValueError(f"Particle filter {name} variances must be nonnegative.")
        return np.sqrt(diag)

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def _ensure_discretization(self, dt: float) -> None:
        dt = float(dt)
        if np.isclose(dt, self._disc_dt, rtol=0.0, atol=1e-12):
            return
        self.A, self.B = discretize_AB(self._plant, dt)
        self._disc_dt = dt

    def _initialize_particles(self, center: np.ndarray) -> None:
        center = np.asarray(center, dtype=float).reshape(8)
        noise = self._rng.normal(0.0, self._init_std, size=(self.num_particles, 8))
        noise -= noise.mean(axis=0)
        self.particles = center + noise
        self.weights.fill(1.0 / self.num_particles)
        self.x_hat = center.reshape(-1, 1)

    def _predict_particles(self, u: np.ndarray) -> None:
        self.particles = self.particles @ self.A.T + (self.B @ u).ravel()
        if np.any(self._process_std > 0.0):
            self.particles += self._rng.normal(
                0.0,
                self._process_std,
                size=self.particles.shape,
            )

    def _normalize_log_weights(self, log_weights: np.ndarray) -> None:
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            self.weights.fill(1.0 / self.num_particles)
            return

        max_log_weight = float(np.max(log_weights[finite]))
        weights = np.zeros_like(self.weights)
        weights[finite] = np.exp(log_weights[finite] - max_log_weight)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.num_particles)
            return

        self.weights = weights / total

    def _update_weights(self, z: np.ndarray) -> None:
        residuals = z.ravel() - self.particles @ self.H.T
        log_likelihood = -0.5 * np.sum(
            (residuals ** 2) / self._measurement_var,
            axis=1,
        )
        log_weights = np.log(np.maximum(self.weights, np.finfo(float).tiny)) + log_likelihood
        self._normalize_log_weights(log_weights)

    def _estimate_mean(self) -> np.ndarray:
        return np.sum(self.weights[:, None] * self.particles, axis=0)

    def _resample(self) -> None:
        indexes = self._rng.choice(
            self.num_particles,
            size=self.num_particles,
            replace=True,
            p=self.weights,
        )
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.num_particles)

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:
        self._ensure_discretization(dt)

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)

        self._predict_particles(u)
        x_pred = self._estimate_mean().reshape(-1, 1)
        innovation = (z - self.H @ x_pred).ravel()

        self._update_weights(z)
        self.x_hat = self._estimate_mean().reshape(-1, 1)

        if self._resample_threshold > 0.0:
            neff_limit = self._resample_threshold * self.num_particles
            if self.effective_sample_size <= neff_limit:
                self._resample()

        return State.from_iterable(self.x_hat.ravel()), innovation

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            center = np.zeros(8, dtype=float)
        else:
            center = x_hat.as_vector()
        self._initialize_particles(center)
