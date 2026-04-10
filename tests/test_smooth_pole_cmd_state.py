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

from src.shared import ControlInput, State, TimingParams, WorkspaceParams, build_from_registry
from src.system.controller import CONTROLLER_REGISTRY
from src.system.controller.smooth_pole import SmoothPoleParams, SmoothPolePlacementController
from src.system.controller.smooth_pole_cmd_state import (
    SMOOTH_POLE_CMD_STATE_PRESETS,
    SmoothPoleCommandStateController,
    SmoothPoleCommandStateParams,
)


def _state(
    *,
    px: float = 0.0,
    vx: float = 0.0,
    ax: float = 0.0,
    wx: float = 0.0,
    py: float = 0.0,
    vy: float = 0.0,
    ay: float = 0.0,
    wy: float = 0.0,
) -> State:
    return State(px=px, vx=vx, ax=ax, wx=wx, py=py, vy=vy, ay=ay, wy=wy)


def _make_controller(*, pos_drift_gain: float = 0.0, dt: float = 0.01) -> SmoothPoleCommandStateController:
    return SmoothPoleCommandStateController(
        SmoothPoleCommandStateParams(
            s_poles=SMOOTH_POLE_CMD_STATE_PRESETS["default"]["s_poles"],
            slew_poles=SMOOTH_POLE_CMD_STATE_PRESETS["default"]["slew_poles"],
            timing=TimingParams(total_time=1.0, dt=dt),
            workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
            pos_drift_gain=pos_drift_gain,
        )
    )


def _set_test_gain(controller: SmoothPoleCommandStateController) -> None:
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 0] = 1.0
    controller.K[0, 1] = 2.0
    controller.K[0, 2] = 3.0
    controller.K[0, 3] = 4.0
    controller.K[0, 8] = 0.5
    controller.K[1, 4] = 1.5
    controller.K[1, 5] = 2.5
    controller.K[1, 6] = 3.5
    controller.K[1, 7] = 4.5
    controller.K[1, 9] = 0.75


def test_smooth_pole_cmd_state_matches_smooth_pole_gain_shape_and_value():
    timing = TimingParams(total_time=1.0, dt=0.01)
    workspace = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None)

    smooth = SmoothPolePlacementController(
        SmoothPoleParams(
            s_poles=SMOOTH_POLE_CMD_STATE_PRESETS["default"]["s_poles"],
            slew_poles=SMOOTH_POLE_CMD_STATE_PRESETS["default"]["slew_poles"],
            timing=timing,
            workspace=workspace,
        )
    )
    cmd_state = SmoothPoleCommandStateController(
        SmoothPoleCommandStateParams(
            s_poles=SMOOTH_POLE_CMD_STATE_PRESETS["default"]["s_poles"],
            slew_poles=SMOOTH_POLE_CMD_STATE_PRESETS["default"]["slew_poles"],
            timing=timing,
            workspace=workspace,
            pos_drift_gain=0.0,
        )
    )

    assert smooth.K.shape == cmd_state.K.shape
    np.testing.assert_allclose(cmd_state.K, smooth.K, atol=1e-12)


def test_smooth_pole_cmd_state_ignores_measured_position_and_velocity_in_main_feedback():
    controller = _make_controller(pos_drift_gain=0.0)
    _set_test_gain(controller)
    controller.set_applied_command(ControlInput(px_cmd=0.1, py_cmd=-0.2))

    base = _state(ax=0.02, wx=-0.03, ay=0.04, wy=-0.05)
    noisy = _state(px=0.5, vx=1.2, ax=0.02, wx=-0.03, py=-0.7, vy=-1.5, ay=0.04, wy=-0.05)

    out_base = controller.compute(base)
    out_noisy = controller.compute(noisy)

    np.testing.assert_allclose(
        [out_base.px_cmd, out_base.py_cmd],
        [out_noisy.px_cmd, out_noisy.py_cmd],
        atol=1e-12,
    )


def test_smooth_pole_cmd_state_responds_to_angle_channels():
    controller = _make_controller(pos_drift_gain=0.0)
    _set_test_gain(controller)
    controller.set_applied_command(ControlInput(px_cmd=0.1, py_cmd=-0.2))

    flat = controller.compute(_state())
    tilted = controller.compute(_state(ax=0.02, wx=-0.03, ay=0.04, wy=-0.05))

    assert not np.allclose(
        [flat.px_cmd, flat.py_cmd],
        [tilted.px_cmd, tilted.py_cmd],
        atol=1e-12,
    )


def test_smooth_pole_cmd_state_updates_command_velocity_from_applied_command():
    controller = _make_controller(pos_drift_gain=0.0, dt=0.01)

    controller.set_applied_command(ControlInput(px_cmd=0.02, py_cmd=-0.01))
    controller.set_applied_command(ControlInput(px_cmd=0.05, py_cmd=0.03))

    np.testing.assert_allclose(controller._u_prev, [0.05, 0.03], atol=1e-12)
    np.testing.assert_allclose(controller._cmd_vel, [3.0, 4.0], atol=1e-12)


def test_smooth_pole_cmd_state_reset_is_bumpless_and_stays_near_estimated_position():
    controller = _make_controller(pos_drift_gain=0.0)
    controller.K = np.zeros((2, 10), dtype=float)
    controller.reset(_state(px=0.03, py=-0.04))

    out = controller.compute(_state())

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.03, -0.04], atol=1e-12)
    np.testing.assert_allclose(controller._u_prev, [0.03, -0.04], atol=1e-12)


def test_smooth_pole_cmd_state_registry_builds():
    controller = build_from_registry(CONTROLLER_REGISTRY, "smooth_pole_cmd_state:default")

    assert isinstance(controller, SmoothPoleCommandStateController)
