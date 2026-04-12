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
class _KalmanModeFilter:
    name: str
    dynamics_mode: str
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x_hat: np.ndarray
    P: np.ndarray


@dataclass
class IMM_KalmanParams:
    q_y_meas_pos: float
    q_y_meas_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_y_meas_pos: float
    r_y_meas_ang: float
    
    q_y_meas_pos_placing: float | None = None
    q_y_meas_ang_placing: float | None = None
    q_vel_pos_placing: float | None = None
    q_vel_ang_placing: float | None = None
    r_y_meas_pos_placing: float | None = None
    r_y_meas_ang_placing: float | None = None
    
    placing_anchor_stiffness: float = 400.0
    placing_anchor_damping: float = 40.0
    placing_gravity_scale: float = 0.0
    mode_stickiness: float = 0.999
    
    free_to_placing_probability: float | None = None
    placing_to_free_probability: float | None = None
    initial_placing_probability: float = 0.5
    min_model_probability: float = 1e-3
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


IMM_KALMAN_PRESETS = {
    "default": {
        "q_y_meas_pos": 1e-8,
        "q_y_meas_ang": 1e-8,
        "q_vel_pos": 1e-7,
        "q_vel_ang": 1e-7,
        "r_y_meas_pos": 1e-5,
        "r_y_meas_ang": 1e-5,
    },
    "real": {
        "base": "default",
        "initial_placing_probability": 0.9,
        "free_to_placing_probability": 0.01,
        "placing_to_free_probability": 0.03,
        "min_model_probability": 1e-3,
        "q_y_meas_ang_placing": 1e-9,
        "q_vel_ang_placing": 1e-8,
        "r_y_meas_ang_placing": 2e-5,
    },
    "test":{
        "q_y_meas_pos": 1e-4,
        "q_y_meas_ang": 1e-4,
        "q_vel_pos": 1e-2,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-3,
        "r_y_meas_ang": 1e-3,
    },
    "test1":{
        "q_y_meas_pos": 1e-4,
        "q_y_meas_ang": 1e-4,
        "q_vel_pos": 1e-1,
        "q_vel_ang": 1e-1,
        "r_y_meas_pos": 1e-3,
        "r_y_meas_ang": 1e-3,
    }
}


class IMM_KalmanEstimator(BaseEstimator):

    def __init__(self, params: IMM_KalmanParams):
        super().__init__()
        p = params
        self._plant = params.plant
        self._disc_dt = float(params.timing.dt)
        self.H = measurement_H()
        self._placing_discretize_kwargs = {
            "placing_anchor_stiffness": float(p.placing_anchor_stiffness),
            "placing_anchor_damping": float(p.placing_anchor_damping),
            "placing_gravity_scale": float(p.placing_gravity_scale),
        }

        self.A, self.B = discretize_AB(self._plant, self._disc_dt, mode="free")
        self.A_placing, self.B_placing = discretize_AB(
            self._plant,
            self._disc_dt,
            mode="placing",
            **self._placing_discretize_kwargs,
        )

        Q_free = self._build_process_noise(
            p.q_y_meas_pos,
            p.q_y_meas_ang,
            p.q_vel_pos,
            p.q_vel_ang,
        )
        R_free = self._build_measurement_noise(
            p.r_y_meas_pos,
            p.r_y_meas_ang,
        )
        Q_placing = self._build_process_noise(
            self._override(p.q_y_meas_pos, p.q_y_meas_pos_placing),
            self._override(p.q_y_meas_ang, p.q_y_meas_ang_placing),
            self._override(p.q_vel_pos, p.q_vel_pos_placing),
            self._override(p.q_vel_ang, p.q_vel_ang_placing),
        )
        R_placing = self._build_measurement_noise(
            self._override(p.r_y_meas_pos, p.r_y_meas_pos_placing),
            self._override(p.r_y_meas_ang, p.r_y_meas_ang_placing),
        )
        self.Q = Q_free
        self.R = R_free

        self._mode_filters = [
            _KalmanModeFilter(
                name="free",
                dynamics_mode="free",
                A=self.A.copy(),
                B=self.B.copy(),
                Q=Q_free.copy(),
                R=R_free.copy(),
                x_hat=np.zeros((8, 1)),
                P=np.eye(8),
            ),
            _KalmanModeFilter(
                name="placing",
                dynamics_mode="placing",
                A=self.A_placing.copy(),
                B=self.B_placing.copy(),
                Q=Q_placing.copy(),
                R=R_placing.copy(),
                x_hat=np.zeros((8, 1)),
                P=np.eye(8),
            ),
        ]

        self.P_init = np.eye(8) * 2e-1
        self.x_hat_init_0 = np.zeros((8, 1))
        self.P = self.P_init.copy()
        self.x_hat = self.x_hat_init_0.copy()

        self._min_model_probability = float(np.clip(p.min_model_probability, 0.0, 0.49))
        self._mode_transition = self._build_mode_transition(
            float(p.mode_stickiness),
            p.free_to_placing_probability,
            p.placing_to_free_probability,
        )
        self._initial_mode_probabilities = self._sanitize_mode_probabilities(
            np.array(
                [
                    1.0 - float(p.initial_placing_probability),
                    float(p.initial_placing_probability),
                ],
                dtype=float,
            )
        )
        self._model_probabilities = self._initial_mode_probabilities.copy()
        self.reset()

    @property
    def model_probabilities(self) -> dict[str, float]:
        return {
            mode.name: float(prob)
            for mode, prob in zip(self._mode_filters, self._model_probabilities)
        }

    @property
    def dominant_model(self) -> str:
        return self._mode_filters[int(np.argmax(self._model_probabilities))].name

    @property
    def adaptive_lpf_weight(self) -> float:
        """Compatibility logging flag: 0 = free model, 1 = placing model."""
        return self.model_probabilities["placing"]

    @staticmethod
    def _override(default: float, override: float | None) -> float:
        if override is None:
            return float(default)
        return float(override)

    @staticmethod
    def _build_process_noise(
        q_y_meas_pos: float,
        q_y_meas_ang: float,
        q_vel_pos: float,
        q_vel_ang: float,
    ) -> np.ndarray:
        return np.diag([
            q_y_meas_pos, q_vel_pos, q_y_meas_ang, q_vel_ang,
            q_y_meas_pos, q_vel_pos, q_y_meas_ang, q_vel_ang,
        ])

    @staticmethod
    def _build_measurement_noise(
        r_y_meas_pos: float,
        r_y_meas_ang: float,
    ) -> np.ndarray:
        return np.diag([
            r_y_meas_pos, r_y_meas_ang,
            r_y_meas_pos, r_y_meas_ang,
        ])

    def _build_mode_transition(
        self,
        mode_stickiness: float,
        free_to_placing_probability: float | None = None,
        placing_to_free_probability: float | None = None,
    ) -> np.ndarray:
        if (
            free_to_placing_probability is not None
            or placing_to_free_probability is not None
        ):
            free_to_placing = float(np.clip(
                self._override(1.0 - mode_stickiness, free_to_placing_probability),
                0.0,
                1.0,
            ))
            placing_to_free = float(np.clip(
                self._override(1.0 - mode_stickiness, placing_to_free_probability),
                0.0,
                1.0,
            ))
            return np.array([
                [1.0 - free_to_placing, free_to_placing],
                [placing_to_free, 1.0 - placing_to_free],
            ], dtype=float)

        stay = float(np.clip(mode_stickiness, 0.5, 1.0))
        switch = 1.0 - stay
        return np.array([
            [stay, switch],
            [switch, stay],
        ], dtype=float)

    def _sanitize_mode_probabilities(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float).reshape(2)
        probs = np.maximum(probs, self._min_model_probability)
        total = float(probs.sum())
        if total <= 0.0:
            return np.array([0.5, 0.5], dtype=float)
        return probs / total

    @staticmethod
    def _logsumexp(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        vmax = float(np.max(values))
        return vmax + float(np.log(np.sum(np.exp(values - vmax))))

    def _innovation_log_likelihood(self, innovation: np.ndarray, S: np.ndarray) -> float:
        S = 0.5 * (S + S.T)
        reg = 1e-12 * np.eye(S.shape[0])
        try:
            S_inv = np.linalg.inv(S)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0.0:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            S = S + reg
            S_inv = np.linalg.pinv(S)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0.0:
                logdet = np.log(max(float(np.linalg.det(S)), 1e-12))

        mahal = float(innovation @ S_inv @ innovation)
        dim = innovation.size
        return -0.5 * (mahal + logdet + dim * np.log(2.0 * np.pi))

    @staticmethod
    def _safe_invert(S: np.ndarray) -> np.ndarray:
        S = 0.5 * (S + S.T)
        try:
            return np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S)

    def _combine_modes(self) -> tuple[np.ndarray, np.ndarray]:
        x_combined = np.zeros((8, 1))
        for prob, mode in zip(self._model_probabilities, self._mode_filters):
            x_combined += prob * mode.x_hat

        P_combined = np.zeros((8, 8))
        for prob, mode in zip(self._model_probabilities, self._mode_filters):
            dx = mode.x_hat - x_combined
            P_combined += prob * (mode.P + dx @ dx.T)
        return x_combined, P_combined

    def _imm_mixing(self) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        mixed_probabilities = self._mode_transition.T @ self._model_probabilities
        mixed_probabilities = self._sanitize_mode_probabilities(mixed_probabilities)

        mixed_states: list[np.ndarray] = []
        mixed_covariances: list[np.ndarray] = []
        for j in range(len(self._mode_filters)):
            denom = max(float(mixed_probabilities[j]), 1e-12)
            mixing_weights = self._mode_transition[:, j] * self._model_probabilities / denom
            total = float(mixing_weights.sum())
            if total <= 0.0:
                mixing_weights = np.full(len(self._mode_filters), 1.0 / len(self._mode_filters))
            else:
                mixing_weights = mixing_weights / total

            x_mix = np.zeros((8, 1))
            for weight, mode in zip(mixing_weights, self._mode_filters):
                x_mix += weight * mode.x_hat

            P_mix = np.zeros((8, 8))
            for weight, mode in zip(mixing_weights, self._mode_filters):
                dx = mode.x_hat - x_mix
                P_mix += weight * (mode.P + dx @ dx.T)

            mixed_states.append(x_mix)
            mixed_covariances.append(P_mix)

        return mixed_probabilities, mixed_states, mixed_covariances

    def _ensure_discretization(self, dt: float) -> None:
        dt = float(dt)
        if np.isclose(dt, self._disc_dt, rtol=0.0, atol=1e-12):
            return
        self.A, self.B = discretize_AB(self._plant, dt, mode="free")
        self.A_placing, self.B_placing = discretize_AB(
            self._plant,
            dt,
            mode="placing",
            **self._placing_discretize_kwargs,
        )
        self._mode_filters[0].A = self.A.copy()
        self._mode_filters[0].B = self.B.copy()
        self._mode_filters[1].A = self.A_placing.copy()
        self._mode_filters[1].B = self.B_placing.copy()
        self._disc_dt = dt

    def _estimate_imm(self, z: np.ndarray, u: np.ndarray) -> tuple[State, np.ndarray]:
        prior_probs, mixed_states, mixed_covariances = self._imm_mixing()
        log_likelihoods = np.zeros(len(self._mode_filters), dtype=float)
        innovations: list[np.ndarray] = []

        for idx, mode in enumerate(self._mode_filters):
            x_mix = mixed_states[idx]
            P_mix = mixed_covariances[idx]

            x_pred = mode.A @ x_mix + mode.B @ u
            innovation = (z - self.H @ x_pred).ravel()
            P_pred = mode.A @ P_mix @ mode.A.T + mode.Q

            S = self.H @ P_pred @ self.H.T + mode.R
            K = P_pred @ self.H.T @ self._safe_invert(S)

            I = np.eye(8)
            I_KH = I - K @ self.H
            mode.x_hat = x_pred + K @ innovation.reshape(-1, 1)
            mode.P = I_KH @ P_pred @ I_KH.T + K @ mode.R @ K.T
            mode.P = 0.5 * (mode.P + mode.P.T)

            innovations.append(innovation)
            log_likelihoods[idx] = self._innovation_log_likelihood(innovation, S)

        log_prior = np.log(np.maximum(prior_probs, 1e-12))
        log_weights = log_prior + log_likelihoods
        log_weights -= self._logsumexp(log_weights)
        self._model_probabilities = self._sanitize_mode_probabilities(np.exp(log_weights))

        self.x_hat, self.P = self._combine_modes()
        innovation = sum(
            prob * residual
            for prob, residual in zip(self._model_probabilities, innovations)
        )
        return State.from_iterable(self.x_hat.flatten()), innovation

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:
        self._ensure_discretization(dt)

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)

        return self._estimate_imm(z, u)

    def reset(self, x_hat: State | None = None):
        if x_hat is not None:
            x_col = x_hat.as_vector().reshape(-1, 1)
        else:
            x_col = self.x_hat_init_0.copy()

        for mode in self._mode_filters:
            mode.x_hat = x_col.copy()
            mode.P = self.P_init.copy()

        self._model_probabilities = self._initial_mode_probabilities.copy()
        self.P = self.P_init.copy()
        self.x_hat = x_col.copy()
