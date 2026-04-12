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
from src.system.controller.simple_controller import (
    SIMPLE_CONTROLLER_PRESETS,
    SimpleController,
    SimpleControllerParams,
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
    kc: float = 0.25,
    kcv: float = 0.02,
    k_alpha: float = 5.0e-3,
    k_omega: float = 5.0e-4,
    max_tilt_ref_deg: float | None = None,
    max_delta_u: float | None = None,
    workspace: WorkspaceParams | None = None,
) -> SimpleController:
    return SimpleController(
        SimpleControllerParams(
            kc=kc,
            kcv=kcv,
            k_alpha=k_alpha,
            k_omega=k_omega,
            max_tilt_ref_deg=max_tilt_ref_deg,
            max_delta_u=max_delta_u,
            workspace=workspace or WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
        )
    )


def test_simple_controller_registry_builds_all_presets():
    for preset in SIMPLE_CONTROLLER_PRESETS:
        controller = build_from_registry(CONTROLLER_REGISTRY, f"simple:{preset}")
        assert isinstance(controller, SimpleController)


def test_simple_controller_zero_state_returns_zero_command():
    controller = _make_controller()

    out = controller.compute(_state())

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.0, 0.0], atol=1e-12)


def test_simple_controller_position_and_velocity_set_tilt_reference_sign():
    controller = _make_controller(kc=2.0, kcv=3.0, k_alpha=5.0, k_omega=0.0)

    out = controller.compute(_state(px=0.1, vx=0.2, py=-0.1, vy=-0.2))

    # alpha_ref_x = -0.62 and alpha_ref_y = +0.62 with signed quadratic position.
    # Positive inner-loop sign means zero actual tilt commands +k_alpha * (0 - alpha_ref).
    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [3.1, -3.1], atol=1e-12)


def test_simple_controller_tilt_and_rate_use_inner_loop_sign():
    controller = _make_controller(kc=0.0, kcv=0.0, k_alpha=2.0, k_omega=3.0)

    out = controller.compute(_state(ax=0.25, wx=-0.5, ay=-0.25, wy=0.5))

    # delta_u = k_alpha * alpha + k_omega * omega when alpha_ref is zero.
    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [-1.0, 1.0], atol=1e-12)


def test_simple_controller_clips_tilt_reference():
    controller = _make_controller(
        kc=100.0,
        kcv=0.0,
        k_alpha=2.0,
        k_omega=0.0,
        max_tilt_ref_deg=1.0,
    )

    out = controller.compute(_state(px=1.0, py=-1.0))

    expected = 2.0 * np.deg2rad(1.0)
    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [expected, -expected], atol=1e-12)


def test_simple_controller_adds_delta_to_previous_applied_command():
    controller = _make_controller(kc=0.0, kcv=0.0, k_alpha=2.0, k_omega=0.0)
    controller.set_applied_command(ControlInput(px_cmd=0.10, py_cmd=-0.20))

    out = controller.compute(_state(ax=0.03, ay=-0.04))

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.16, -0.28], atol=1e-12)


def test_simple_controller_limits_delta_u_from_applied_command():
    controller = _make_controller(
        kc=0.0,
        kcv=0.0,
        k_alpha=1.0,
        k_omega=0.0,
        max_delta_u=0.5,
    )
    controller.set_applied_command(ControlInput(px_cmd=1.0, py_cmd=1.0))

    out = controller.compute(_state(ax=-2.0, ay=1.0))
    delta = np.array([out.px_cmd - 1.0, out.py_cmd - 1.0])

    np.testing.assert_allclose(np.linalg.norm(delta), 0.5, atol=1e-12)
    np.testing.assert_allclose(delta / np.linalg.norm(delta), np.array([-2.0, 1.0]) / np.sqrt(5.0))


def test_simple_controller_reset_uses_workspace_reference():
    controller = _make_controller(
        kc=0.0,
        kcv=0.0,
        k_alpha=1.0,
        k_omega=0.0,
        max_delta_u=0.5,
        workspace=WorkspaceParams(x_ref=0.02, y_ref=-0.03, safe_radius=None),
    )
    controller.set_applied_command(ControlInput(px_cmd=1.0, py_cmd=1.0))

    controller.reset(_state(px=0.5, py=0.5))
    out = controller.compute(_state())

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.02, -0.03], atol=1e-12)
    np.testing.assert_allclose(controller._u_prev, [0.02, -0.03], atol=1e-12)
