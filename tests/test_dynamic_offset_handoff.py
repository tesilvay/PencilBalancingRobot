import numpy as np

from src.shared import (
    ControlInput,
    Measurement,
    State,
    StepData,
    TableAccel,
    WorkspaceParams,
)
from src.system.system import System, SystemParams
from src.system.supervisor.dynamic import DynamicSupervisor, DynamicSupervisorParams
from src.experiment.logger.logger import Logger, LoggerParams


def _state(px: float, py: float) -> State:
    return State(px=px, vx=0.0, ax=0.0, wx=0.0, py=py, vy=0.0, ay=0.0, wy=0.0)


class _PlantSequence:
    def __init__(self, states: list[State]):
        self.states = list(states)
        self.i = 0
        self.last_in: State | None = None
        self.top_radius = None

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        self.last_in = state_x
        out = self.states[min(self.i, len(self.states) - 1)]
        self.i += 1
        return out, TableAccel(0.0, 0.0)

    def set_top_radius(self, radius: float) -> None:
        self.top_radius = float(radius)

    def reset(self):
        self.i = 0
        self.last_in = None


class _Sensor:
    def __init__(self):
        self.last_y: Measurement | None = None

    def get_y(self, x_true: State) -> Measurement:
        y = Measurement(px=x_true.px, py=x_true.py, ax=x_true.ax, ay=x_true.ay)
        self.last_y = y
        return y

    def reset(self):
        self.last_y = None


class _Estimator:
    def __init__(self):
        self.last_y_meas: Measurement | None = None
        self.reset_args: list[State | None] = []

    def estimate(self, y_meas: Measurement, dt: float, u_cmd: ControlInput | None):
        del dt, u_cmd
        self.last_y_meas = y_meas
        x_hat = State(px=y_meas.px, vx=0.0, ax=y_meas.ax, wx=0.0, py=y_meas.py, vy=0.0, ay=y_meas.ay, wy=0.0)
        return x_hat, np.zeros(4)

    def reset(self, x_hat: State | None = None):
        self.reset_args.append(x_hat)


class _Controller:
    def __init__(self):
        self.reset_args: list[State | None] = []
        self.last_applied: ControlInput | None = None

    def compute(self, state: State) -> ControlInput:
        del state
        return ControlInput(0.0, 0.0)

    def set_applied_command(self, u: ControlInput, state: State | None = None) -> None:
        del state
        self.last_applied = u

    def reset(self, x_hat: State | None = None):
        self.reset_args.append(x_hat)


class _Actuator:
    def mech_joint_snapshot(self, command) -> np.ndarray:
        del command
        return np.full((3, 2), np.nan, dtype=float)

    def apply(self, command) -> np.ndarray:
        return self.mech_joint_snapshot(command)


class _GainSchedule:
    def apply(self, y: Measurement) -> Measurement:
        return y

    def reset(self):
        pass


class _SupervisorScripted:
    def __init__(self, steps: list[tuple[int, int, dict | None]]):
        self.steps = steps
        self.i = 0
        self._last_transition = None

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        del x_est, innovation, dt
        ctrl_i, est_i, tr = self.steps[min(self.i, len(self.steps) - 1)]
        self._last_transition = tr
        self.i += 1
        return ctrl_i, est_i

    @property
    def last_transition(self):
        return self._last_transition


def test_dynamic_supervisor_ramps_top_radius_after_acquisition():
    sup = DynamicSupervisor(
        DynamicSupervisorParams(
            radius_ramp_s=5.0,
            placing_time_s=1.0,
            min_radius=0.0,
            max_radius=None,
        )
    )
    sup.attach_runtime(workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.07))

    assert sup.state_name == "ACQUISITION"
    assert sup.top_radius == 0.0

    ctrl_i, est_k = sup.update(_state(0.0, 0.0), np.zeros(4), _state(0.0, 0.0), np.zeros(4), dt=0.01)
    assert (ctrl_i, est_k) == (0, 0.0)
    assert sup.last_transition is not None
    assert sup.last_transition["prev_state"] == "ACQUISITION"
    assert sup.last_transition["new_state"] == "ACQUISITION"
    assert sup.last_transition["left_acquisition"] is False
    assert sup.is_offset_latched is False
    assert sup.top_radius == 0.0

    sup.update(_state(0.0, 0.0), np.zeros(4), _state(0.0, 0.0), np.zeros(4), dt=0.99)
    assert sup.state_name == "STABILIZATION"
    assert sup.last_transition is not None
    assert sup.last_transition["prev_state"] == "ACQUISITION"
    assert sup.last_transition["new_state"] == "STABILIZATION"
    assert sup.last_transition["left_acquisition"] is True
    assert sup.is_offset_latched is True
    assert sup.top_radius == 0.0

    sup.update(_state(0.0, 0.0), np.zeros(4), _state(0.0, 0.0), np.zeros(4), dt=2.5)
    np.testing.assert_allclose(sup.top_radius, 0.035, atol=1e-12)

    sup.update(_state(0.0, 0.0), np.zeros(4), _state(0.0, 0.0), np.zeros(4), dt=2.5)
    np.testing.assert_allclose(sup.top_radius, 0.07, atol=1e-12)


def test_system_applies_dynamic_top_radius_without_switching_components():
    plant_acq = _PlantSequence([_state(0.03, -0.02), _state(0.04, -0.01)])
    plant_sim = _PlantSequence([_state(0.08, 0.05)])
    c0, c1 = _Controller(), _Controller()
    e0, e1 = _Estimator(), _Estimator()
    sup = DynamicSupervisor(
        DynamicSupervisorParams(
            radius_ramp_s=1.0,
            placing_time_s=1.0,
            min_radius=0.0,
            max_radius=0.05,
        )
    )
    system = System(
        SystemParams(
            plants=[plant_acq, plant_sim],
            controllers=[c0, c1],
            estimators=[e0, e1],
            sensor=_Sensor(),
            actuator=_Actuator(),
            supervisor=sup,
            gain_schedule=_GainSchedule(),
            workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
        )
    )
    system.reset()

    system.step(0.5)
    assert system.active_plant is plant_acq
    assert system.active_controller is c0
    assert system.active_estimator is e0
    np.testing.assert_allclose(plant_acq.top_radius, 0.0, atol=1e-12)
    np.testing.assert_allclose(plant_sim.top_radius, 0.0, atol=1e-12)
    np.testing.assert_allclose(system.step_data.offset_xy, [0.03, -0.02], atol=1e-12)
    assert system.step_data.offset_latched is False

    system.step(0.5)
    assert system.active_plant is plant_acq
    assert system.active_controller is c0
    assert system.active_estimator is e0
    np.testing.assert_allclose(plant_acq.top_radius, 0.0, atol=1e-12)
    np.testing.assert_allclose(plant_sim.top_radius, 0.0, atol=1e-12)
    np.testing.assert_allclose(system.step_data.offset_xy, [0.04, -0.01], atol=1e-12)
    assert system.step_data.offset_latched is True

    system.step(0.5)
    np.testing.assert_allclose(system.step_data.top_radius, 0.0, atol=1e-12)
    np.testing.assert_allclose(plant_acq.top_radius, 0.025, atol=1e-12)
    np.testing.assert_allclose(plant_sim.top_radius, 0.025, atol=1e-12)

    system.step(0.01)
    np.testing.assert_allclose(system.step_data.top_radius, 0.025, atol=1e-12)
    np.testing.assert_allclose(system.step_data.offset_xy, [0.04, -0.01], atol=1e-12)
    assert system.step_data.offset_latched is True


def test_system_applies_static_workspace_safe_radius():
    from src.system.supervisor.static import StaticSupervisor, StaticSupervisorParams

    plant = _PlantSequence([_state(0.01, 0.02)])
    system = System(
        SystemParams(
            plants=[plant],
            controllers=[_Controller()],
            estimators=[_Estimator()],
            sensor=_Sensor(),
            actuator=_Actuator(),
            supervisor=StaticSupervisor(StaticSupervisorParams()),
            gain_schedule=_GainSchedule(),
            workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.068),
        )
    )
    system.reset()
    np.testing.assert_allclose(plant.top_radius, 0.068, atol=1e-12)


def test_logger_records_offset_history_and_latched_flags():
    logger = Logger(LoggerParams())
    step0 = StepData(
        x=_state(0.0, 0.0),
        u=ControlInput(0.0, 0.0),
        acc=TableAccel(0.0, 0.0),
        innovation=np.zeros(4),
        mech_joints=np.full((3, 2), np.nan),
        offset_xy=np.array([0.01, -0.02]),
        offset_latched=False,
        top_radius=0.01,
    )
    logger.reset(step0)
    logger.record(
        StepData(
            x=_state(0.0, 0.0),
            u=ControlInput(0.0, 0.0),
            acc=TableAccel(0.0, 0.0),
            innovation=np.zeros(4),
            mech_joints=np.full((3, 2), np.nan),
            offset_xy=np.array([0.03, -0.01]),
            offset_latched=True,
            top_radius=0.04,
        )
    )
    result = logger.get_result()
    assert result.offset_history is not None
    assert result.offset_latched_history is not None
    assert result.offset_history.shape == (2, 2)
    assert result.offset_latched_history.shape == (2,)
    np.testing.assert_allclose(result.offset_history[0], [0.01, -0.02], atol=1e-12)
    np.testing.assert_allclose(result.offset_history[1], [0.03, -0.01], atol=1e-12)
    assert bool(result.offset_latched_history[0]) is False
    assert bool(result.offset_latched_history[1]) is True
    assert result.top_radius_history is not None
    np.testing.assert_allclose(result.top_radius_history, [0.01, 0.04], atol=1e-12)
