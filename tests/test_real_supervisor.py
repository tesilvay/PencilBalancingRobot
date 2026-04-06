import numpy as np

from src.shared import ControlInput, Measurement, State, TableAccel, WorkspaceParams
from src.system.system import System, SystemParams
from src.system.supervisor.real_dynamic import RealDynamicSupervisor, RealDynamicSupervisorParams
from src.system.supervisor.real import RealSupervisor, RealSupervisorParams


def _state(px: float, py: float, *, ax: float = 0.0, ay: float = 0.0) -> State:
    return State(px=px, vx=0.0, ax=ax, wx=0.0, py=py, vy=0.0, ay=ay, wy=0.0)


class _PlantSequence:
    def __init__(self, states: list[State]):
        self.states = list(states)
        self.i = 0

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        del state_x, u_cmd, dt
        out = self.states[min(self.i, len(self.states) - 1)]
        self.i += 1
        return out, TableAccel(0.0, 0.0)

    def reset(self):
        self.i = 0


class _Sensor:
    def get_y(self, x_true: State) -> Measurement:
        return Measurement(px=x_true.px, py=x_true.py, ax=x_true.ax, ay=x_true.ay)

    def reset(self):
        pass


class _Estimator:
    def __init__(self):
        self.last_y_meas: Measurement | None = None
        self.reset_args: list[State | None] = []

    def estimate(self, y_meas: Measurement, dt: float, u_cmd: ControlInput | None):
        del dt, u_cmd
        self.last_y_meas = y_meas
        return _state(y_meas.px, y_meas.py, ax=y_meas.ax, ay=y_meas.ay), np.zeros(4)

    def reset(self, x_hat: State | None = None):
        self.reset_args.append(x_hat)


class _Controller:
    def __init__(self, out: ControlInput):
        self.out = out
        self.last_applied: ControlInput | None = None
        self.reset_args: list[State | None] = []

    def compute(self, state: State) -> ControlInput:
        del state
        return self.out

    def set_applied_command(self, u: ControlInput) -> None:
        self.last_applied = u

    def reset(self, x_hat: State | None = None):
        self.reset_args.append(x_hat)


class _Actuator:
    def __init__(self):
        self.commands: list[ControlInput] = []
        self.offsets: list[tuple[float, float]] = []

    def mech_joint_snapshot(self, command) -> np.ndarray:
        del command
        return np.full((3, 2), np.nan, dtype=float)

    def apply(self, command) -> np.ndarray:
        self.commands.append(command)
        return self.mech_joint_snapshot(command)

    def set_workspace_offset(self, dx: float, dy: float) -> None:
        self.offsets.append((float(dx), float(dy)))

    def reset(self):
        pass


def _build_system(
    supervisor: RealSupervisor,
    *,
    states: list[State],
    workspace: WorkspaceParams,
) -> tuple[System, _Actuator, _Controller, _Controller]:
    c0 = _Controller(ControlInput(-0.02, 0.01))
    c1 = _Controller(ControlInput(0.03, -0.04))
    actuator = _Actuator()
    system = System(
        SystemParams(
            plants=[_PlantSequence(states), _PlantSequence(states)],
            controllers=[c0, c1],
            estimators=[_Estimator()],
            sensor=_Sensor(),
            actuator=actuator,
            supervisor=supervisor,
            workspace=workspace,
        )
    )
    system.reset()
    return system, actuator, c0, c1


def test_real_supervisor_starts_in_centering_and_handles_manual_keys():
    workspace = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.003)
    sup = RealSupervisor(
        RealSupervisorParams(
            centering_controller_index=0,
            run_controller_index=1,
            estimator_index=0,
            stable_threshold_deg=3.0,
            stable_hold_s=0.2,
            manual_step_m=0.002,
            workspace=workspace,
        )
    )
    actuator = _Actuator()
    sup.attach_runtime(actuator=actuator, workspace=workspace)
    sup.reset()

    assert sup.state_name == "SERVO_CENTERING"
    assert sup.command_override is not None
    assert actuator.offsets[-1] == (0.0, 0.0)

    assert sup.handle_key(ord("d")) is True
    assert sup.handle_key(ord("w")) is True
    np.testing.assert_allclose(
        [sup.manual_target.px_cmd, sup.manual_target.py_cmd],
        [0.002, 0.002],
        atol=1e-12,
    )

    assert sup.handle_key(ord("d")) is True
    scale = workspace.safe_radius / np.hypot(0.004, 0.002)
    np.testing.assert_allclose(
        [sup.manual_target.px_cmd, sup.manual_target.py_cmd],
        [0.004 * scale, 0.002 * scale],
        atol=1e-12,
    )

    assert sup.handle_key(13) is True
    assert sup.state_name == "ACQUISITION"
    assert sup.command_override is not None
    np.testing.assert_allclose(
        [sup.command_override.px_cmd, sup.command_override.py_cmd],
        [workspace.x_ref, workspace.y_ref],
        atol=1e-12,
    )
    assert sup.is_offset_latched is False
    np.testing.assert_allclose(actuator.offsets[-1], [0.004 * scale, 0.002 * scale], atol=1e-12)


def test_real_supervisor_transitions_to_balanced_without_estimator_switch():
    workspace = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None)
    sup = RealSupervisor(
        RealSupervisorParams(
            centering_controller_index=0,
            run_controller_index=1,
            estimator_index=0,
            stable_threshold_deg=3.0,
            stable_hold_s=0.2,
            manual_step_m=0.001,
            workspace=workspace,
        )
    )
    sup.attach_runtime(actuator=_Actuator(), workspace=workspace)
    sup.reset()
    sup.handle_key(13)

    idx0 = sup.update(_state(0.0, 0.0, ax=0.01, ay=-0.01), np.zeros(4), 0.1)
    idx1 = sup.update(_state(0.0, 0.0, ax=0.01, ay=-0.01), np.zeros(4), 0.1)

    assert idx0 == (0, 0)
    assert idx1 == (1, 0)
    assert sup.state_name == "BALANCED"
    assert sup.is_offset_latched is True
    assert sup.last_transition is not None
    assert sup.last_transition["prev_state"] == "ACQUISITION"
    assert sup.last_transition["new_state"] == "BALANCED"


def test_system_uses_supervisor_command_override_then_returns_to_controller():
    workspace = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None)
    sup = RealSupervisor(
        RealSupervisorParams(
            centering_controller_index=0,
            run_controller_index=1,
            estimator_index=0,
            stable_threshold_deg=3.0,
            stable_hold_s=0.05,
            manual_step_m=0.002,
            workspace=workspace,
        )
    )
    system, actuator, c0, c1 = _build_system(
        sup,
        states=[
            _state(0.0, 0.0, ax=0.1, ay=0.0),
            _state(0.0, 0.0, ax=0.1, ay=0.0),
            _state(0.0, 0.0, ax=0.0, ay=0.0),
            _state(0.0, 0.0, ax=0.0, ay=0.0),
        ],
        workspace=workspace,
    )

    sup.handle_key(ord("d"))
    system.step(0.01)
    np.testing.assert_allclose(
        [actuator.commands[-1].px_cmd, actuator.commands[-1].py_cmd],
        [0.002, 0.0],
        atol=1e-12,
    )
    assert c0.last_applied is not None
    np.testing.assert_allclose([c0.last_applied.px_cmd, c0.last_applied.py_cmd], [0.002, 0.0], atol=1e-12)

    sup.handle_key(13)
    system.step(0.01)
    np.testing.assert_allclose(
        [actuator.offsets[-1][0], actuator.offsets[-1][1]],
        [0.002, 0.0],
        atol=1e-12,
    )
    assert system.step_data.offset_latched is False
    np.testing.assert_allclose(
        [actuator.commands[-1].px_cmd, actuator.commands[-1].py_cmd],
        [workspace.x_ref, workspace.y_ref],
        atol=1e-12,
    )

    system.step(0.05)
    system.step(0.01)
    system.step(0.01)
    assert system.step_data.offset_latched is True
    np.testing.assert_allclose(
        [actuator.commands[-1].px_cmd, actuator.commands[-1].py_cmd],
        [c1.out.px_cmd, c1.out.py_cmd],
        atol=1e-12,
    )
    assert c1.reset_args


def test_real_dynamic_switches_to_next_estimator_after_delay():
    workspace = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None)
    sup = RealDynamicSupervisor(
        RealDynamicSupervisorParams(
            centering_controller_index=0,
            run_controller_index=1,
            acquisition_estimator_index=0,
            run_estimator_index=1,
            stable_threshold_deg=3.0,
            stable_hold_s=0.05,
            estimator_switch_delay_s=0.2,
            manual_step_m=0.002,
            workspace=workspace,
        )
    )
    sup.attach_runtime(actuator=_Actuator(), workspace=workspace)
    sup.reset()
    sup.handle_key(13)

    idx0 = sup.update(_state(0.0, 0.0, ax=0.1, ay=0.0), np.zeros(4), 0.01)
    idx1 = sup.update(_state(0.0, 0.0, ax=0.0, ay=0.0), np.zeros(4), 0.05)

    assert idx0 == (0, 0)
    assert idx1 == (1, 0)
    assert sup.state_name == "STABILIZING"
    assert sup.is_offset_latched is True
    idx2 = sup.update(_state(0.0, 0.0, ax=0.0, ay=0.0), np.zeros(4), 0.1)
    assert idx2 == (1, 0)
    idx3 = sup.update(_state(0.0, 0.0, ax=0.0, ay=0.0), np.zeros(4), 0.1)
    assert idx3 == (1, 1)
    assert sup.state_name == "BALANCED"
