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
class AugKalmanParams:
    q_y_meas_pos: float
    q_y_meas_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_y_meas_pos: float
    r_y_meas_ang: float

    # New: disturbance-state tuning
    q_dist_ang: float = 5e-1        # process noise for fingertip disturbance states
    dist_decay: float = 0.995       # leak per sample; <1 means disturbance fades after release
    p_init_state: float = 2e-1
    p_init_dist: float = 5.0

    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    
AUG_KALMAN_PRESETS = {
    "default": {
        "q_y_meas_pos": 1e-6,
        "q_y_meas_ang": 1e-6,
        "q_vel_pos": 1e-3,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-2,
        "r_y_meas_ang": 7e-2,
        
        "q_dist_ang": 5e-1, # How fast the disturbance estimate is allowed to move.
        "dist_decay": 0.99, # How fast the disturbance disappears after release.
        "p_init_state": 2e-1,
        "p_init_dist": 5.0,
    },
    "finger_hold": {
        "q_y_meas_pos": 1e-6,
        "q_y_meas_ang": 1e-6,
        "q_vel_pos": 1e-3,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-2,
        "r_y_meas_ang": 7e-2,
        "q_dist_ang": 2.0,
        "dist_decay": 0.99,
        "p_init_state": 2e-1,
        "p_init_dist": 20.0,
    },
}

class AugmentedKalmanEstimator(BaseEstimator):
    """
    10-state Kalman filter with two augmented disturbance states:
        d_x, d_y = unknown angular-acceleration disturbances
    These capture the fingertip support torque during startup.

    State ordering:
        [x, x_dot, ax, ax_dot, y, y_dot, ay, ay_dot, d_x, d_y]^T

    Measurement ordering stays:
        [x, ax, y, ay]^T
    """

    def __init__(self, params: AugKalmanParams):
        super().__init__()
        self._params = params
        self._dt = params.timing.dt

        A8, B8 = discretize_AB(params.plant, params.timing)
        H4x8 = measurement_H()

        self.A = self._build_augmented_A(A8, params.dist_decay, self._dt)
        self.B = self._build_augmented_B(B8)
        self.H = self._build_augmented_H(H4x8)

        self.Q = self._build_process_covariance(params)
        self.R = self._build_measurement_covariance(params)

        self.P_init = self._build_initial_covariance(params)
        self.x_hat_init_0 = np.zeros((10, 1))

        self.P = self.P_init.copy()
        self.x_hat = self.x_hat_init_0.copy()

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:

        z = self.measurement_z(y_meas)              # shape (4,)
        u = self.control_u(u_cmd, y_meas)           # expected shape compatible with B

        x_pred = self._predict_state(self.x_hat, u)
        P_pred = self._predict_covariance(self.P)

        innovation = self._compute_innovation(z, x_pred)
        K, S = self._compute_kalman_gain(P_pred)

        self.x_hat = x_pred + K @ innovation.reshape(-1, 1)
        self.P = self._update_covariance_joseph(P_pred, K)

        x_hat_state = self._extract_state(self.x_hat)
        return x_hat_state, innovation

    def reset(self, x_hat: State | None = None):
        self.P = self.P_init.copy()
        self.x_hat = self._build_reset_state(x_hat)

    def disturbance_estimate(self) -> tuple[float, float]:
        return float(self.x_hat[8, 0]), float(self.x_hat[9, 0])

    def _build_augmented_A(
        self,
        A8: np.ndarray,
        dist_decay: float,
        dt: float,
    ) -> np.ndarray:
        A = np.zeros((10, 10))
        A[:8, :8] = A8

        # Disturbance states represent additive angular acceleration.
        # Discrete-time effect:
        #   angle_{k+1}     += 0.5 * dt^2 * d
        #   angle_dot_{k+1} += dt * d
        Gd = np.zeros((8, 2))

        # x-plane angular states: ax, ax_dot are indices 2, 3
        Gd[2, 0] = 0.5 * dt * dt
        Gd[3, 0] = dt

        # y-plane angular states: ay, ay_dot are indices 6, 7
        Gd[6, 1] = 0.5 * dt * dt
        Gd[7, 1] = dt

        A[:8, 8:] = Gd
        A[8:, 8:] = np.eye(2) * dist_decay
        return A

    def _build_augmented_B(self, B8: np.ndarray) -> np.ndarray:
        B = np.zeros((10, B8.shape[1]))
        B[:8, :] = B8
        return B

    def _build_augmented_H(self, H4x8: np.ndarray) -> np.ndarray:
        H = np.zeros((H4x8.shape[0], 10))
        H[:, :8] = H4x8
        return H

    def _build_process_covariance(self, p: AugKalmanParams) -> np.ndarray:
        return np.diag([
            p.q_y_meas_pos, p.q_vel_pos, p.q_y_meas_ang, p.q_vel_ang,
            p.q_y_meas_pos, p.q_vel_pos, p.q_y_meas_ang, p.q_vel_ang,
            p.q_dist_ang, p.q_dist_ang,
        ])

    def _build_measurement_covariance(self, p: AugKalmanParams) -> np.ndarray:
        return np.diag([
            p.r_y_meas_pos, p.r_y_meas_ang,
            p.r_y_meas_pos, p.r_y_meas_ang,
        ])

    def _build_initial_covariance(self, p: AugKalmanParams) -> np.ndarray:
        P = np.eye(10) * p.p_init_state
        P[8, 8] = p.p_init_dist
        P[9, 9] = p.p_init_dist
        return P

    def _build_reset_state(self, x_hat: State | None) -> np.ndarray:
        x = np.zeros((10, 1))
        if x_hat is not None:
            x[:8, 0] = x_hat.as_vector().flatten()
        return x

    def _predict_state(self, x_hat: np.ndarray, u: np.ndarray) -> np.ndarray:
        u_col = np.asarray(u).reshape(-1, 1)
        return self.A @ x_hat + self.B @ u_col

    def _predict_covariance(self, P: np.ndarray) -> np.ndarray:
        return self.A @ P @ self.A.T + self.Q

    def _compute_innovation(self, z: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        return z.reshape(-1, 1).flatten() - (self.H @ x_pred).flatten()

    def _compute_kalman_gain(self, P_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S = self.H @ P_pred @ self.H.T + self.R
        K = np.linalg.solve(S.T, (P_pred @ self.H.T).T).T
        return K, S

    def _update_covariance_joseph(self, P_pred: np.ndarray, K: np.ndarray) -> np.ndarray:
        I = np.eye(self.A.shape[0])
        KH = K @ self.H
        return (I - KH) @ P_pred @ (I - KH).T + K @ self.R @ K.T

    def _extract_state(self, x_hat: np.ndarray) -> State:
        return State.from_iterable(x_hat[:8, 0])