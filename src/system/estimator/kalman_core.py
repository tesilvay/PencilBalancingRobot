from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanStepResult:
    x_hat: np.ndarray
    P: np.ndarray
    K: np.ndarray
    S: np.ndarray
    y: np.ndarray
    nis: float


def _nis_from_innovation(S: np.ndarray, y: np.ndarray) -> float:
    sol = np.linalg.solve(S, y)
    return float((y.T @ sol).item())


def run_linear_kalman_step(
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x_hat: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
) -> KalmanStepResult:
    x_pred = A @ x_hat + B @ u
    P_pred = A @ P @ A.T + Q
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    y = z - H @ x_pred
    x_new = x_pred + K @ y
    I = np.eye(x_hat.shape[0])
    P_new = (I - K @ H) @ P_pred
    nis = _nis_from_innovation(S, y)
    return KalmanStepResult(
        x_hat=x_new,
        P=P_new,
        K=K,
        S=S,
        y=y,
        nis=nis,
    )
