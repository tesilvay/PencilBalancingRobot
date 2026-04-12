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

from src.shared import ControlInput, Measurement, TimingParams, default_plant
from src.system.estimator.dynamics_disc import _continuous_AB, discretize_AB
from src.system.estimator.imm_kalman import IMM_KalmanEstimator, IMM_KalmanParams


def _imm_params(dt: float) -> IMM_KalmanParams:
    return IMM_KalmanParams(
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


def test_placing_dynamics_include_soft_anchor_terms():
    plant = default_plant()
    k_anchor = 625.0
    c_anchor = 50.0
    gravity_scale = 0.25
    A, B = _continuous_AB(
        plant,
        mode="placing",
        placing_anchor_stiffness=k_anchor,
        placing_anchor_damping=c_anchor,
        placing_gravity_scale=gravity_scale,
    )

    alpha_coeff = gravity_scale * plant.g / plant.com_length - k_anchor
    assert np.isclose(A[2, 3], 1.0)
    assert np.isclose(A[3, 2], alpha_coeff)
    assert np.isclose(A[3, 3], -c_anchor)
    assert np.isclose(A[6, 7], 1.0)
    assert np.isclose(A[7, 6], alpha_coeff)
    assert np.isclose(A[7, 7], -c_anchor)
    assert np.isclose(B[3, 0], -1.0 / (plant.com_length * plant.tau**2))
    assert np.isclose(B[7, 1], -1.0 / (plant.com_length * plant.tau**2))


def test_imm_prefers_placing_model_for_placing_like_response():
    dt = 3e-3
    estimator = IMM_KalmanEstimator(_imm_params(dt))
    A_place, B_place = discretize_AB(
        estimator._plant,
        dt,
        mode="placing",
        **estimator._placing_discretize_kwargs,
    )

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
    assert np.isclose(estimator.adaptive_lpf_weight, probs["placing"])


def test_imm_prefers_free_model_for_balancing_like_response():
    dt = 5e-3
    estimator = IMM_KalmanEstimator(_imm_params(dt))
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
    assert np.isclose(estimator.adaptive_lpf_weight, probs["placing"])


def test_imm_joseph_update_keeps_covariance_well_behaved():
    dt = 1e-3
    estimator = IMM_KalmanEstimator(_imm_params(dt))
    command = ControlInput(0.0, 0.0)
    meas = Measurement(
        px=0.0,
        py=0.0,
        ax=float(np.deg2rad(3.0)),
        ay=float(np.deg2rad(-2.0)),
    )

    for _ in range(150):
        estimator.estimate(meas, dt=dt, u_cmd=command)

    assert np.all(np.isfinite(estimator.P))
    assert np.allclose(estimator.P, estimator.P.T)
    assert np.all(np.diag(estimator.P) >= -1e-12)
    for mode in estimator._mode_filters:
        assert np.all(np.isfinite(mode.P))
        assert np.allclose(mode.P, mode.P.T)
        assert np.all(np.diag(mode.P) >= -1e-12)
