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

from src.shared import ControlInput, State, build_from_registry
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
    g_p: float = 0.25,
    g_a: float = 0.02,
    g_d: float = 5.0e-3,
) -> SimpleController:
    return SimpleController(SimpleControllerParams(g_p=g_p, g_a=g_a, g_d=g_d))


def test_simple_controller_registry_builds_all_presets():
    for preset in SIMPLE_CONTROLLER_PRESETS:
        controller = build_from_registry(CONTROLLER_REGISTRY, f"simple:{preset}")
        assert isinstance(controller, SimpleController)


def test_simple_controller_zero_state_returns_zero_command():
    controller = _make_controller()

    out = controller.compute(_state())

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.0, 0.0], atol=1e-12)


def test_simple_controller_position_tilt_and_velocity_set_command():
    controller = _make_controller(g_p=2.0, g_a=3.0, g_d=5.0)

    out = controller.compute(
        _state(px=0.1, ax=0.3, vx=0.02, py=-0.1, ay=-0.3, vy=-0.02)
    )

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [1.2, -1.2], atol=1e-12)


def test_simple_controller_reference_command_is_zero():
    controller = _make_controller(g_p=2.0, g_a=3.0, g_d=5.0)

    out = controller.reference_command()

    np.testing.assert_allclose([out.px_cmd, out.py_cmd], [0.0, 0.0], atol=1e-12)


def test_simple_controller_is_stateless():
    controller = _make_controller(g_p=0.0, g_a=2.0, g_d=0.0)
    state = _state(ax=0.03, ay=-0.04)

    before = controller.compute(state)
    controller.set_applied_command(ControlInput(px_cmd=0.10, py_cmd=-0.20), state)
    controller.reset(state)
    after = controller.compute(state)

    np.testing.assert_allclose([before.px_cmd, before.py_cmd], [0.06, -0.08], atol=1e-12)
    np.testing.assert_allclose([after.px_cmd, after.py_cmd], [0.06, -0.08], atol=1e-12)
