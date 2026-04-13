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

from src.shared import ControlInput, State, TimingParams, WorkspaceParams, build_from_registry
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
    actuator_dt: float = 0.003,
    tilt_stale_time_s: float = 0.5,
    tilt_deadband: float = np.deg2rad(0.1),
    tilt_ki: float = 0.05,
    max_tilt_bias: float = np.deg2rad(2.0),
) -> DeltaLQRController:
    return DeltaLQRController(
        DeltaLQRParams(
            pos_scale=1e-2,
            vel_scale=3e-2,
            tilt_scale=np.deg2rad(2.0),
            tilt_rate_scale=np.deg2rad(10.0),
            delta_u_scale=1e-3,
            q_pos=1.0,
            q_vel=1.0,
            q_tilt=1.0,
            q_tilt_rate=1.0,
            q_command=0.0,
            r_delta_u=1.0,
            pos_thresh=1.5e-2,
            angle_thresh=np.deg2rad(3.6),
            rate_thresh=None,
            ki=0.4,
            tilt_stale_time_s=tilt_stale_time_s,
            tilt_deadband=tilt_deadband,
            tilt_ki=tilt_ki,
            max_tilt_bias=max_tilt_bias,
            max_delta_u=max_delta_u,
            max_command_radius=max_command_radius,
            timing=TimingParams(total_time=10.0, dt=1e-3, actuator_dt=actuator_dt),
            workspace=workspace or WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
        )
    )


def _apply_ticks(controller: DeltaLQRController, state: State, ticks: int) -> None:
    for _ in range(ticks):
        controller.set_applied_command(ControlInput(px_cmd=0.0, py_cmd=0.0), state)


def test_delta_lqr_registry_builds_all_presets():
    for preset in DELTA_LQR_PRESETS:
        controller = build_from_registry(CONTROLLER_REGISTRY, f"delta_lqr:{preset}")
        assert isinstance(controller, DeltaLQRController)


def test_delta_lqr_zero_gain_holds_previous_applied_command():
    controller = _make_controller()
    controller.set_applied_command(ControlInput(px_cmd=0.02, py_cmd=-0.03), _state())

    out = controller.compute(_state(px=1.0, py=-1.0, ax=0.5, ay=-0.5))

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.02, -0.03], atol=1e-12)


def test_delta_lqr_adds_lqr_delta_to_previous_applied_command():
    controller = _make_controller()
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 0] = -2.0
    controller.K[1, 4] = 3.0
    controller.set_applied_command(ControlInput(px_cmd=0.10, py_cmd=-0.20), _state())

    out = controller.compute(_state(px=0.03, py=-0.04))

    # delta_u = -K @ [x_err; u_err].
    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.16, -0.08], atol=1e-12)


def test_delta_lqr_limits_delta_u_not_absolute_command():
    controller = _make_controller(max_delta_u=0.5)
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 2] = -2.0
    controller.K[1, 6] = 1.0
    controller.set_applied_command(ControlInput(px_cmd=1.0, py_cmd=1.0), _state())

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


def test_delta_lqr_tilt_reference_waits_for_stale_angle():
    controller = _make_controller(actuator_dt=0.1)
    stale_state = _state(px=0.02, ax=np.deg2rad(0.5))

    _apply_ticks(controller, stale_state, 4)
    np.testing.assert_allclose(controller._x_ref_lqr[[2, 6]], [0.0, 0.0], atol=1e-12)

    _apply_ticks(controller, stale_state, 1)
    assert controller._tilt_x_integrator_active
    assert controller._x_ref_lqr[2] < 0.0
    np.testing.assert_allclose(controller._x_ref_lqr[6], 0.0, atol=1e-12)


def test_delta_lqr_tilt_reference_resets_stale_timer_on_deadband_and_sign_flip():
    controller = _make_controller(actuator_dt=0.1)

    _apply_ticks(controller, _state(px=0.02, ax=np.deg2rad(0.5)), 4)
    _apply_ticks(controller, _state(px=0.02, ax=0.0), 1)
    _apply_ticks(controller, _state(px=0.02, ax=np.deg2rad(0.5)), 4)
    np.testing.assert_allclose(controller._x_ref_lqr[2], 0.0, atol=1e-12)

    _apply_ticks(controller, _state(px=0.02, ax=-np.deg2rad(0.5)), 1)
    _apply_ticks(controller, _state(px=0.02, ax=-np.deg2rad(0.5)), 3)
    np.testing.assert_allclose(controller._x_ref_lqr[2], 0.0, atol=1e-12)

    _apply_ticks(controller, _state(px=0.02, ax=-np.deg2rad(0.5)), 1)
    assert controller._x_ref_lqr[2] > 0.0


def test_delta_lqr_tilt_reference_latches_after_stale_angle():
    controller = _make_controller(actuator_dt=0.1)

    _apply_ticks(controller, _state(px=0.02, ax=np.deg2rad(0.5)), 5)
    ref_after_stale = controller._x_ref_lqr[2]
    assert controller._tilt_x_integrator_active
    assert ref_after_stale < 0.0

    _apply_ticks(controller, _state(px=0.0, ax=np.deg2rad(0.5)), 1)
    assert controller._x_ref_lqr[2] < ref_after_stale


def test_delta_lqr_tilt_reference_uses_its_own_latch_not_position_latch():
    controller = _make_controller(actuator_dt=0.1)

    _apply_ticks(controller, _state(px=0.0, ax=np.deg2rad(0.5)), 5)
    assert controller._tilt_x_integrator_active
    assert not controller._integrator_active
    assert controller._x_ref_lqr[2] < 0.0


def test_delta_lqr_tilt_reference_ignores_angle_and_rate_gates_when_stale():
    controller = _make_controller(actuator_dt=0.1)
    stale_state = _state(px=0.0, ax=np.deg2rad(4.0), wx=np.deg2rad(100.0))

    _apply_ticks(controller, stale_state, 5)

    assert not controller._integrator_active
    assert controller._x_ref_lqr[2] < 0.0


def test_delta_lqr_tilt_reference_is_clamped():
    max_tilt_ref = np.deg2rad(0.25)
    controller = _make_controller(
        actuator_dt=0.1,
        tilt_ki=10.0,
        max_tilt_bias=max_tilt_ref,
    )

    _apply_ticks(controller, _state(px=0.02, ax=np.deg2rad(0.5), ay=-np.deg2rad(0.5)), 10)

    np.testing.assert_allclose(
        controller._x_ref_lqr[[2, 6]],
        [-max_tilt_ref, max_tilt_ref],
        atol=1e-12,
    )


def test_delta_lqr_tilt_reference_affects_lqr_error_and_u_ref():
    controller = _make_controller(actuator_dt=0.1)
    controller.K = np.zeros((2, 10), dtype=float)
    controller.K[0, 2] = -2.0

    _apply_ticks(controller, _state(px=0.02, ax=np.deg2rad(0.5)), 5)
    ref_ax = controller._x_ref_lqr[2]
    assert ref_ax < 0.0

    out = controller.compute(_state())

    np.testing.assert_allclose(out.px_cmd, 2.0 * -ref_ax, atol=1e-12)
    np.testing.assert_allclose(controller.u_ref_lqr, (controller._u_ref_true @ controller._x_ref_lqr).ravel())


def test_delta_lqr_reset_clears_tilt_reference_integrator():
    controller = _make_controller(actuator_dt=0.1)
    _apply_ticks(controller, _state(px=0.02, ax=np.deg2rad(0.5)), 5)
    assert controller._x_ref_lqr[2] < 0.0

    controller.reset()

    np.testing.assert_allclose(controller._x_ref_lqr[[2, 6]], [0.0, 0.0], atol=1e-12)
    assert not controller._tilt_x_integrator_active
    assert not controller._tilt_y_integrator_active
    np.testing.assert_allclose(controller._tilt_x_same_sign_time, 0.0, atol=1e-12)
    np.testing.assert_allclose(controller._tilt_y_same_sign_time, 0.0, atol=1e-12)
    np.testing.assert_allclose(controller._tilt_x_prev_sign, 0.0, atol=1e-12)
    np.testing.assert_allclose(controller._tilt_y_prev_sign, 0.0, atol=1e-12)
