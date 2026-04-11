import sys
import types
from pathlib import Path

import numpy as np


def _install_control_stub():
    control = sys.modules.get("control")
    if control is not None and getattr(control, "__file__", None):
        return

    control = types.ModuleType("control") if control is None else control

    def ss(A, B, C, D):
        return types.SimpleNamespace(A=np.array(A), B=np.array(B), C=np.array(C), D=np.array(D))

    def c2d(sys, dt):
        A = np.array(sys.A, dtype=float)
        B = np.array(sys.B, dtype=float)
        n = A.shape[0]
        return types.SimpleNamespace(
            A=np.eye(n, dtype=float) + A * float(dt),
            B=B * float(dt),
        )

    def ctrb(A, B):
        A = np.array(A)
        B = np.array(B)
        n = A.shape[0]
        return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])

    def obsv(A, C):
        A = np.array(A)
        C = np.array(C)
        n = A.shape[0]
        return np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])

    control.ss = ss
    control.c2d = c2d
    control.ctrb = ctrb
    control.obsv = obsv
    sys.modules["control"] = control


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_install_control_stub()

from src.shared import ControlInput, Measurement, TimingParams
from src.system.estimator.dynamics_disc import discretize_AB
from src.system.estimator.kalman import KalmanEstimator, KalmanParams


def _kalman_params(dt: float) -> KalmanParams:
    return KalmanParams(
        q_y_meas_pos=1e-8,
        q_y_meas_ang=1e-8,
        q_vel_pos=1e-6,
        q_vel_ang=1e-6,
        r_y_meas_pos=1e-5,
        r_y_meas_ang=1e-5,
        timing=TimingParams(total_time=1.0, dt=dt, actuator_dt=dt),
        mode_stickiness=0.995,
        initial_placing_probability=0.5,
        min_model_probability=1e-4,
    )


def _measurement_from_state(state_vec: np.ndarray) -> Measurement:
    x = np.asarray(state_vec, dtype=float).reshape(-1)
    return Measurement(
        px=float(x[0]),
        ax=float(x[2]),
        py=float(x[4]),
        ay=float(x[6]),
    )


def test_kalman_prefers_placing_model_for_placing_like_response():
    dt = 3e-3
    estimator = KalmanEstimator(_kalman_params(dt))
    A_place, B_place = discretize_AB(estimator._plant, dt, mode="placing")

    x = np.array([
        [0.0],
        [0.0],
        [np.deg2rad(4.0)],
        [0.0],
        [0.0],
        [0.0],
        [np.deg2rad(-3.0)],
        [0.0],
    ], dtype=float)
    u = np.zeros((2, 1), dtype=float)
    command = ControlInput(0.0, 0.0)

    for _ in range(120):
        estimator.estimate(_measurement_from_state(x), dt=dt, u_cmd=command)
        x = A_place @ x + B_place @ u

    probs = estimator.model_probabilities
    assert probs["placing"] > 0.9
    assert probs["free"] < 0.1


def test_kalman_prefers_free_model_for_balancing_like_response():
    dt = 5e-3
    estimator = KalmanEstimator(_kalman_params(dt))
    A_free, B_free = discretize_AB(estimator._plant, dt, mode="free")

    x = np.array([
        [0.0],
        [0.0],
        [np.deg2rad(2.0)],
        [0.0],
        [0.0],
        [0.0],
        [np.deg2rad(-1.5)],
        [0.0],
    ], dtype=float)
    u = np.zeros((2, 1), dtype=float)
    command = ControlInput(0.0, 0.0)

    for _ in range(120):
        estimator.estimate(_measurement_from_state(x), dt=dt, u_cmd=command)
        x = A_free @ x + B_free @ u

    probs = estimator.model_probabilities
    assert probs["free"] > 0.9
    assert probs["placing"] < 0.1


def test_kalman_stationary_tilt_converges_without_runaway_table_drift():
    dt = 1e-3
    estimator = KalmanEstimator(_kalman_params(dt))
    command = ControlInput(0.0, 0.0)
    meas = Measurement(
        px=0.0,
        py=0.0,
        ax=float(np.deg2rad(3.0)),
        ay=float(np.deg2rad(-2.0)),
    )

    x_hat = None
    for _ in range(150):
        x_hat, _ = estimator.estimate(meas, dt=dt, u_cmd=command)

    probs = estimator.model_probabilities
    assert probs["placing"] > 0.8
    assert x_hat is not None
    assert abs(x_hat.px) < 2e-3
    assert abs(x_hat.py) < 2e-3
    assert abs(x_hat.vx) < 2e-2
    assert abs(x_hat.vy) < 2e-2
