import sys
import types
from pathlib import Path

import numpy as np


def _install_control_stub():
    if "control" in sys.modules:
        return

    control = types.ModuleType("control")

    def ss(A, B, C, D):
        return types.SimpleNamespace(A=np.array(A), B=np.array(B), C=np.array(C), D=np.array(D))

    def c2d(sys, dt):
        del dt
        return types.SimpleNamespace(A=np.array(sys.A), B=np.array(sys.B))

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

from src.shared import Measurement, TimingParams, build_from_registry
from src.system.estimator import ESTIMATOR_REGISTRY
from src.system.estimator.lpf import LPFParams, LowPassFiniteDifferenceEstimator


def test_lpf_registry_builds():
    estimator = build_from_registry(ESTIMATOR_REGISTRY, "lpf:default")

    assert isinstance(estimator, LowPassFiniteDifferenceEstimator)


def test_lpf_auto_history_matches_sense_to_actuate_ratio():
    estimator = LowPassFiniteDifferenceEstimator(
        LPFParams(
            alpha_meas=0.0,
            alpha_vel=0.0,
            timing=TimingParams(total_time=1.0, dt=0.5e-3, actuator_dt=5e-3),
        )
    )

    assert estimator.history_size == 10


def test_lpf_history_window_reduces_gaussian_measurement_noise():
    rng = np.random.default_rng(7)
    true_meas = np.array([0.02, -0.01, -0.03, 0.015], dtype=float)
    noise_std = np.array([3e-3, 2e-3, 3e-3, 2e-3], dtype=float)

    estimator = LowPassFiniteDifferenceEstimator(
        LPFParams(
            alpha_meas=0.0,
            alpha_vel=0.0,
            history_size=10,
            timing=TimingParams(total_time=1.0, dt=0.5e-3, actuator_dt=5e-3),
        )
    )

    raw_err = []
    filt_err = []

    for _ in range(200):
        sample = true_meas + rng.normal(0.0, noise_std)
        meas = Measurement(px=sample[0], py=sample[2], ax=sample[1], ay=sample[3])
        x_hat, _ = estimator.estimate(meas, dt=0.5e-3, u_cmd=None)

        raw_err.append(sample - true_meas)
        filt_err.append(np.array([x_hat.px, x_hat.ax, x_hat.py, x_hat.ay]) - true_meas)

    raw_err = np.asarray(raw_err[20:], dtype=float)
    filt_err = np.asarray(filt_err[20:], dtype=float)

    assert filt_err.std(axis=0).mean() < raw_err.std(axis=0).mean()


def test_lpf_history_window_recovers_velocity_better_than_two_point_difference():
    rng = np.random.default_rng(11)
    dt = 0.5e-3
    true_vel = np.array([0.08, -0.12, -0.05, 0.09], dtype=float)
    noise_std = np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=float)

    lpf = LowPassFiniteDifferenceEstimator(
        LPFParams(
            alpha_meas=0.0,
            alpha_vel=0.0,
            history_size=10,
            timing=TimingParams(total_time=1.0, dt=dt, actuator_dt=5e-3),
        )
    )
    fde_like = LowPassFiniteDifferenceEstimator(
        LPFParams(
            alpha_meas=0.0,
            alpha_vel=0.0,
            history_size=2,
            timing=TimingParams(total_time=1.0, dt=dt, actuator_dt=5e-3),
        )
    )

    vel_err_lpf = []
    vel_err_fde = []

    for k in range(200):
        t = k * dt
        clean = true_vel * t
        sample = clean + rng.normal(0.0, noise_std)
        meas = Measurement(px=sample[0], py=sample[2], ax=sample[1], ay=sample[3])
        x_lpf, _ = lpf.estimate(meas, dt=dt, u_cmd=None)
        x_fde, _ = fde_like.estimate(meas, dt=dt, u_cmd=None)

        vel_lpf = np.array([x_lpf.vx, x_lpf.wx, x_lpf.vy, x_lpf.wy])
        vel_fde = np.array([x_fde.vx, x_fde.wx, x_fde.vy, x_fde.wy])
        vel_err_lpf.append(vel_lpf - true_vel)
        vel_err_fde.append(vel_fde - true_vel)

    vel_err_lpf = np.asarray(vel_err_lpf[20:], dtype=float)
    vel_err_fde = np.asarray(vel_err_fde[20:], dtype=float)

    rms_lpf = np.sqrt(np.mean(vel_err_lpf ** 2))
    rms_fde = np.sqrt(np.mean(vel_err_fde ** 2))

    assert rms_lpf < rms_fde
