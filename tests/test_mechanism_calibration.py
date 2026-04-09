from pathlib import Path
import sys
import types

import numpy as np


def _install_control_stub():
    if "control" in sys.modules:
        return

    control = types.ModuleType("control")

    def ss(A, B, C, D):
        return types.SimpleNamespace(A=np.array(A), B=np.array(B), C=np.array(C), D=np.array(D))

    def c2d(sys_obj, dt):
        del dt
        return types.SimpleNamespace(A=np.array(sys_obj.A), B=np.array(sys_obj.B))

    def place(A, B, poles):
        del poles
        return np.zeros((np.array(B).shape[1], np.array(A).shape[0]))

    def lqr(A, B, Q, R):
        del Q, R
        return np.zeros((np.array(B).shape[1], np.array(A).shape[0])), None, None

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
    control.place = place
    control.lqr = lqr
    control.ctrb = ctrb
    control.obsv = obsv
    sys.modules["control"] = control


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_install_control_stub()

from src.shared import ControlInput
from src.system.actuator.mech import Mechanism, MechanismParams


def _make_mechanism(tmp_path: Path) -> Mechanism:
    return Mechanism(
        MechanismParams(
            O=np.array([128.77, 178.13]),
            B=np.array([101.77, 210.13]),
            la=175,
            lb=175,
            calibration_file=str(tmp_path / "five_bar_calibration.json"),
        )
    )


def test_mechanism_calibration_targets_prefer_saved_servo_points(tmp_path):
    mechanism = _make_mechanism(tmp_path)
    mechanism.save_calibration(
        {
            "center": {
                "desired_xy_m": np.array([0.0, 0.0]),
                "servo_xy_m": np.array([0.001, -0.002]),
            },
            "up": {
                "desired_xy_m": np.array([0.0, 0.015]),
                "servo_xy_m": np.array([0.002, 0.014]),
            },
            "right": {
                "desired_xy_m": np.array([0.015, 0.0]),
                "servo_xy_m": np.array([0.016, -0.001]),
            },
            "down": {
                "desired_xy_m": np.array([0.0, -0.015]),
                "servo_xy_m": np.array([0.001, -0.016]),
            },
            "left": {
                "desired_xy_m": np.array([-0.015, 0.0]),
                "servo_xy_m": np.array([-0.014, -0.001]),
            },
        }
    )

    targets = mechanism.calibration_targets(
        x_ref=0.0,
        y_ref=0.0,
        cardinal_delta_m=0.015,
        safe_radius=0.068,
    )

    by_name = {target["name"]: target for target in targets}
    np.testing.assert_allclose(by_name["center"]["seed_servo_xy_m"], [0.001, -0.002])
    np.testing.assert_allclose(by_name["up"]["seed_servo_xy_m"], [0.002, 0.014])


def test_mechanism_calibration_targets_use_full_safe_radius_for_cardinals(tmp_path):
    mechanism = _make_mechanism(tmp_path)

    targets = mechanism.calibration_targets(
        x_ref=0.010,
        y_ref=-0.020,
        cardinal_delta_m=0.015,
        safe_radius=0.068,
    )

    by_name = {target["name"]: target for target in targets}
    np.testing.assert_allclose(by_name["center"]["desired_xy_m"], [0.010, -0.020])
    np.testing.assert_allclose(by_name["up"]["desired_xy_m"], [0.010, 0.048])
    np.testing.assert_allclose(by_name["down"]["desired_xy_m"], [0.010, -0.088])
    np.testing.assert_allclose(by_name["right"]["desired_xy_m"], [0.078, -0.020])
    np.testing.assert_allclose(by_name["left"]["desired_xy_m"], [-0.058, -0.020])


def test_mechanism_saved_affine_calibration_is_applied_to_commands(tmp_path):
    mechanism = _make_mechanism(tmp_path)
    mechanism.save_calibration(
        {
            "center": {
                "desired_xy_m": np.array([0.0, 0.0]),
                "servo_xy_m": np.array([0.010, -0.020]),
            },
            "up": {
                "desired_xy_m": np.array([0.0, 0.010]),
                "servo_xy_m": np.array([0.010, -0.009]),
            },
            "right": {
                "desired_xy_m": np.array([0.010, 0.0]),
                "servo_xy_m": np.array([0.021, -0.020]),
            },
            "down": {
                "desired_xy_m": np.array([0.0, -0.010]),
                "servo_xy_m": np.array([0.010, -0.031]),
            },
            "left": {
                "desired_xy_m": np.array([-0.010, 0.0]),
                "servo_xy_m": np.array([-0.001, -0.020]),
            },
        }
    )

    joints, _ = mechanism.command_geometry(ControlInput(px_cmd=0.005, py_cmd=-0.004))
    expected_servo_xy_m = np.array([0.0155, -0.0244])
    np.testing.assert_allclose(joints[2] / 1000.0, expected_servo_xy_m, atol=1e-6)
