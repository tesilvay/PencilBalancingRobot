import numpy as np

from src.shared import (
    ControlInput,
    Measurement,
    State,
    StepData,
    TableAccel,
    WorkspaceParams,
    NullParams,
)
from src.system.system import System, SystemParams
from src.system.supervisor.dynamic import DynamicSupervisor, DynamicSupervisorParams
from src.experiment.logger.logger import Logger


def _state(px: float, py: float) -> State:
    return State(px=px, vx=0.0, ax=0.0, wx=0.0, py=py, vy=0.0, ay=0.0, wy=0.0)


class _PlantSequence:
    def __init__(self, states: list[State]):
        self.states = list(states)
        self.i = 0
        self.last_in: State | None = None

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        self.last_in = state_x
        out = self.states[min(self.i, len(self.states) - 1)]
        self.i += 1
        return out, TableAccel(0.0, 0.0)

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

    def set_applied_command(self, u: ControlInput) -> None:
        self.last_applied = u

    def reset(self, x_hat: State | None = None):
        self.reset_args.append(x_hat)


class _Actuator:
    def mech_joint_snapshot(self, command) -> np.ndarray:
        del command
        return np.full((3, 2), np.nan, dtype=float)

    def apply(self, command) -> np.ndarray:
        return self.mech_joint_snapshot(command)


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


def test_dynamic_supervisor_emits_left_acquisition_transition():
    sup = DynamicSupervisor(
        DynamicSupervisorParams(
            stable_threshold=1.0,
            stable_hold_s=0.1,
            consistent_hold_s=1.0,
            loss_threshold=100.0,
            loss_hold_s=1.0,
        )
    )
    ctrl_i, est_i = sup.update(_state(0.0, 0.0), np.zeros(4), dt=0.11)
    assert (ctrl_i, est_i) == (1, 0)
    assert sup.last_transition is not None
    assert sup.last_transition["prev_state"] == "ACQUISITION"
    assert sup.last_transition["new_state"] == "STABILIZATION_READY"
    assert sup.last_transition["left_acquisition"] is True


def test_system_latches_offset_on_left_acquisition_and_keeps_state_continuous():
    plant_acq = _PlantSequence([_state(0.03, -0.02), _state(0.04, -0.01)])
    plant_sim = _PlantSequence([_state(0.08, 0.05)])
    c0, c1 = _Controller(), _Controller()
    e0, e1 = _Estimator(), _Estimator()
    sup = _SupervisorScripted(
        [
            (0, 0, {"prev_state": "ACQUISITION", "new_state": "ACQUISITION", "left_acquisition": False}),
            (1, 1, {"prev_state": "ACQUISITION", "new_state": "STABILIZATION_READY", "left_acquisition": True}),
            (1, 1, {"prev_state": "STABILIZATION_READY", "new_state": "STABILIZING", "left_acquisition": False}),
        ]
    )
    system = System(
        SystemParams(
            plants=[plant_acq, plant_sim],
            controllers=[c0, c1],
            estimators=[e0, e1],
            sensor=_Sensor(),
            actuator=_Actuator(),
            supervisor=sup,
            workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
        )
    )
    system.reset()

    system.step(0.01)
    np.testing.assert_allclose(system.step_data.offset_xy, [0.03, -0.02], atol=1e-12)
    assert system.step_data.offset_latched is False
    np.testing.assert_allclose(e0.last_y_meas.as_vector()[[0, 2]], [0.0, 0.0], atol=1e-12)

    system.step(0.01)
    np.testing.assert_allclose(system.step_data.offset_xy, [0.04, -0.01], atol=1e-12)
    assert system.step_data.offset_latched is True

    # First sim step receives previous true state from placing at handoff.
    system.step(0.01)
    assert plant_sim.last_in is not None
    assert plant_sim.last_in.px == 0.04
    assert plant_sim.last_in.py == -0.01
    np.testing.assert_allclose(system.step_data.offset_xy, [0.04, -0.01], atol=1e-12)
    assert system.step_data.offset_latched is True


def test_switches_warm_start_new_controller_and_estimator_with_xhat():
    plant0 = _PlantSequence([_state(0.01, 0.02)])
    plant1 = _PlantSequence([_state(0.01, 0.02)])
    c0, c1 = _Controller(), _Controller()
    e0, e1 = _Estimator(), _Estimator()
    sup = _SupervisorScripted(
        [
            (1, 1, {"prev_state": "ACQUISITION", "new_state": "STABILIZATION_READY", "left_acquisition": True}),
        ]
    )
    system = System(
        SystemParams(
            plants=[plant0, plant1],
            controllers=[c0, c1],
            estimators=[e0, e1],
            sensor=_Sensor(),
            actuator=_Actuator(),
            supervisor=sup,
            workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=None),
        )
    )
    system.reset()
    system.step(0.01)

    assert len(c1.reset_args) == 1
    assert len(e1.reset_args) == 1
    assert c1.reset_args[0] is not None
    assert e1.reset_args[0] is not None


def test_logger_records_offset_history_and_latched_flags():
    logger = Logger(NullParams())
    step0 = StepData(
        x=_state(0.0, 0.0),
        u=ControlInput(0.0, 0.0),
        acc=TableAccel(0.0, 0.0),
        innovation=np.zeros(4),
        mech_joints=np.full((3, 2), np.nan),
        offset_xy=np.array([0.01, -0.02]),
        offset_latched=False,
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
