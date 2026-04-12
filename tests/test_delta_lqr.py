import sys
from pathlib import Path

import numpy as np


def _install_control_stub():
    if "control" in sys.modules:
        return

    import types

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

    def dlqr(A, B, Q, R):
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
    control.dlqr = dlqr
    control.ctrb = ctrb
    control.obsv = obsv
    sys.modules["control"] = control


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_install_control_stub()

from src.shared import ControlInput, State, WorkspaceParams, build_from_registry
from src.system.controller import CONTROLLER_REGISTRY
from src.system.controller.delta_lqr import (
    DELTA_LQR_PRESETS,
    DeltaLQRController,
    DeltaLQRParams,
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


def _make_controller(
    *,
    max_delta_u: float | None = None,
    max_command_radius: float | None = None,
    workspace: WorkspaceParams | None = None,
) -> DeltaLQRController:
    return DeltaLQRController(
        DeltaLQRParams(
            q_pos=1.0,
            q_vel=1.0,
            q_tilt=1.0,
            q_tilt_rate=1.0,
            q_command=0.0,
            r_delta_u=1.0,
            max_delta_u=max_delta_u,
            max_command_radius=max_command_radius,
            workspace=workspace or WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
        )
    )


def test_delta_lqr_registry_builds_all_presets():
    for preset in DELTA_LQR_PRESETS:
        controller = build_from_registry(CONTROLLER_REGISTRY, f"delta_lqr:{preset}")
        assert isinstance(controller, DeltaLQRController)


def test_delta_lqr_zero_gain_holds_previous_applied_command():
    controller = _make_controller()
    controller.set_applied_command(ControlInput(px_cmd=0.02, py_cmd=-0.03))

    out = controller.compute(_state(px=1.0, py=-1.0, ax=0.5, ay=-0.5))

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.02, -0.03], atol=1e-12)


def test_delta_lqr_adds_lqr_delta_to_previous_applied_command():
    controller = _make_controller()
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 0] = -2.0
    controller.K[1, 4] = 3.0
    controller.set_applied_command(ControlInput(px_cmd=0.10, py_cmd=-0.20))

    out = controller.compute(_state(px=0.03, py=-0.04))

    # delta_u = -K @ [x_err; u_err].
    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.16, -0.08], atol=1e-12)


def test_delta_lqr_limits_delta_u_not_absolute_command():
    controller = _make_controller(max_delta_u=0.5)
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 2] = -2.0
    controller.K[1, 6] = 1.0
    controller.set_applied_command(ControlInput(px_cmd=1.0, py_cmd=1.0))

    out = controller.compute(_state(ax=1.0, ay=1.0))
    delta = np.array([out.px_cmd - 1.0, out.py_cmd - 1.0])

    np.testing.assert_allclose(np.linalg.norm(delta), 0.5, atol=1e-12)
    np.testing.assert_allclose(delta / np.linalg.norm(delta), np.array([2.0, -1.0]) / np.sqrt(5.0))


def test_delta_lqr_command_radius_clamps_final_absolute_command():
    controller = _make_controller(
        max_command_radius=0.1,
        workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
    )
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 2] = -1.0
    controller.K[1, 6] = -1.0

    out = controller.compute(_state(ax=1.0, ay=1.0))

    np.testing.assert_allclose(np.linalg.norm([out.px_cmd, out.py_cmd]), 0.1, atol=1e-12)


def test_delta_lqr_reset_with_state_is_bumpless():
    controller = _make_controller()
    controller.K = np.zeros((2, 10), dtype=float)
    controller.reset(_state(px=0.1, py=-0.1))

    out = controller.compute(_state(px=0.2, py=-0.2))

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(controller._u_prev, [0.0, 0.0], atol=1e-12)
