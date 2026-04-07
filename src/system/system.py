from dataclasses import dataclass, field

import numpy as np

from src.shared import (
    NullParams,
    State,
    Measurement,
    InitConditionsSpread,
    default_spread,
    ControlInput,
    StepData,
    TableAccel,
    WorkspaceParams,
    clamp_control_input_to_workspace,
    default_workspace,
)

@dataclass
class SystemParams:
    plants:       list
    # Ordered lists of collaborators sharing a common base type. Supervisors
    # select active instances by integer index; order must match preset + supervisor.
    controllers: list
    estimators:  list
    sensor:      object
    actuator:    object
    supervisor:  object
    init_spread: InitConditionsSpread = field(default_factory=default_spread)
    workspace:   WorkspaceParams     = field(default_factory=default_workspace)
    fall_angle_deg: float            = 20.0
    fall_hold_s: float               = 0.03


SYSTEM_PRESETS = {
    "simple_sim": {
        "plants":       ["sim:default"],
        "controllers": ["smooth_pole:smoother"],
        "estimators":  ["lpf:test"],
        "sensor":      "sim_analytic:noisy",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
    },
    "placing_only": {
        "plants":       ["sim:default"],#["placing:angle_only"],
        "controllers": ["smooth_pole:default"],
        "estimators":  ["kalman:default"],
        "sensor":      "sim_dvs:hough",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
    },
    "dynamic_sim": {
        "plants":       ["placing:steady_hands", "sim:default"],
        "controllers": ["null:default", "smooth_pole:default"],
        "estimators":  ["lpf:default", "kalman:default"],
        "sensor":      "sim_analytic:default",
        "actuator":    "mock:default",
        "supervisor":  "dynamic:default",
    },
    "real_vision": {
        "base": "simple_sim",
        "estimators":  ["lpf:test"],
        "sensor":   "real_dvs:hough",
    },
    "real": {
        "base": "real_vision",
        "actuator": "servo:default",
    },
    "real_supervised": {
        "base": "real_vision",
        "plants": ["sim:default", "sim:default"],
        "controllers": ["null:default", "smooth_pole:smoother"],
        "estimators":  ["lpf:smoother"],
        "actuator": "servo:default",
        "supervisor": "real:default",
    },
    "real_dynamic_supervised": {
        "base": "real_vision",
        "plants": ["sim:default", "sim:default"],
        "controllers": ["null:default", "smooth_pole:smoother"],
        "estimators": ["lpf:smoother", "kalman:test"],
        "actuator": "servo:default",
        "supervisor": "real_dynamic:default",
    },
}



class System:
    def __init__(self, params: SystemParams):
        self.plants      = params.plants
        self.controllers = params.controllers
        self.estimators  = params.estimators
        self.sensor      = params.sensor
        self.actuator    = params.actuator
        self.supervisor  = params.supervisor
        self.init_spread = params.init_spread
        self.workspace   = params.workspace
        
        self.active_plant  = self.plants[0]
        self.active_controller = self.controllers[0]
        self.active_estimator  = self.estimators[0]
        
        
        self.step_data: StepData | None = None
        self.x = None
        self.u = None
        self.last_y_meas = None
        self._offset_xy = np.zeros(2, dtype=float)
        self._offset_latched_fallback = False
        self.fall_angle_rad = float(np.deg2rad(params.fall_angle_deg))
        self.fall_hold_s = float(params.fall_hold_s)
        self._fall_detected = False
        self._fall_timer_s = 0.0
        if hasattr(self.supervisor, "attach_runtime"):
            self.supervisor.attach_runtime(actuator=self.actuator, workspace=self.workspace)

    def _offset_from_state(self, x_true: State) -> np.ndarray:
        return np.array(
            [
                float(x_true.px - self.workspace.x_ref),
                float(x_true.py - self.workspace.y_ref),
            ],
            dtype=float,
        )
    
    def _offset_from_meas(self, y: Measurement) -> np.ndarray:
        return np.array(
            [
                float(y.px - self.workspace.x_ref),
                float(y.py - self.workspace.y_ref),
            ],
            dtype=float,
        )

    def _measurement_with_offset(self, y: Measurement, offset_xy: np.ndarray) -> Measurement:
        return Measurement(
            px=float(y.px - offset_xy[0]),
            py=float(y.py - offset_xy[1]),
            ax=float(y.ax),
            ay=float(y.ay),
        )

    def finalize_command(self, u_raw: ControlInput) -> ControlInput:
        u_applied = clamp_control_input_to_workspace(u_raw, self.workspace)
        self.active_controller.set_applied_command(u_applied)
        return u_applied

    def _supervisor_active_indices(self) -> tuple[int, int]:
        try:
            active_indices = self.supervisor.active_indices
        except (AttributeError, NotImplementedError):
            ctrl_i = self.controllers.index(self.active_controller)
            est_i = self.estimators.index(self.active_estimator)
            return ctrl_i, est_i
        return active_indices

    def _is_offset_latched(self) -> bool:
        if hasattr(self.supervisor, "is_offset_latched"):
            return bool(self.supervisor.is_offset_latched)
        return self._offset_latched_fallback

    def _sync_active_components(self, ctrl_i: int, est_i: int, x_hat: State | None = None) -> None:
        new_estimator = self.estimators[est_i]
        if new_estimator is not self.active_estimator:
            new_estimator.reset(x_hat)

        new_controller = self.controllers[ctrl_i]
        if new_controller is not self.active_controller:
            new_controller.reset(x_hat)

        self.active_plant = self.plants[ctrl_i]
        self.active_controller = new_controller
        self.active_estimator = new_estimator

    @property
    def fall_detected(self) -> bool:
        return self._fall_detected

    def _reset_fall_detection(self) -> None:
        self._fall_detected = False
        self._fall_timer_s = 0.0

    def _update_fall_detection(self, y_meas: Measurement, dt: float) -> bool:
        if getattr(self.supervisor, "is_prestart_state", False):
            self._reset_fall_detection()
            return False

        angle_norm = float(np.linalg.norm([y_meas.ax, y_meas.ay]))
        if angle_norm >= self.fall_angle_rad:
            self._fall_timer_s += dt
        else:
            self._fall_timer_s = 0.0
            self._fall_detected = False

        prev_fall_detected = self._fall_detected
        if self._fall_timer_s >= self.fall_hold_s:
            self._fall_detected = True

        return (not prev_fall_detected) and self._fall_detected

    def _print_fall_diagnostic(self, y_meas: Measurement) -> None:
        angle_norm_deg = float(np.rad2deg(np.linalg.norm([y_meas.ax, y_meas.ay])))
        threshold_deg = float(np.rad2deg(self.fall_angle_rad))
        hold_progress = 1.0 if self.fall_hold_s <= 0.0 else min(self._fall_timer_s / self.fall_hold_s, 1.0)

        print(
            "fall: "
            f"|a|={angle_norm_deg:6.2f} deg, "
            f"thresh={threshold_deg:5.2f} deg | "
            f"hold={self._fall_timer_s:5.3f}/{self.fall_hold_s:5.3f} s "
            f"({hold_progress * 100:5.1f}%) | "
            f"latched={self._fall_detected}"
        )

    def step(self, dt):
        prev_supervisor_state = getattr(self.supervisor, "state_name", None)
        self._sync_active_components(*self._supervisor_active_indices())

        x_true, acc = self.active_plant.step(self.x, self.u, dt)

        # get measurements
        y = self.sensor.get_y(x_true)
        self.last_y_meas = y
        
        # no latch, keep updating offset
        if not self._is_offset_latched():
            self._offset_xy = self._offset_from_meas(y)
        y = self._measurement_with_offset(y, self._offset_xy)

        # every estimator calculates innovation too
        x_hat, innovation = self.active_estimator.estimate(y_meas=y, dt=dt, u_cmd=self.u)
        u_raw = self.active_controller.compute(x_hat)
        u_override = getattr(self.supervisor, "command_override", None)
        if u_override is not None:
            u_raw = u_override
        u_cmd = self.finalize_command(u_raw)

        mech_joints = self.actuator.apply(u_cmd)
        if self._update_fall_detection(y, dt):
            self.supervisor.notify_fall_detected()
        self._print_fall_diagnostic(y)

        # 2. supervisor decides what should be active next step
        ctrl_i, est_i = self.supervisor.update(x_hat, innovation, dt)
        new_supervisor_state = getattr(self.supervisor, "state_name", None)
        if prev_supervisor_state == "ACQUISITION" and new_supervisor_state != "ACQUISITION":
            self._reset_fall_detection()
        transition = getattr(self.supervisor, "last_transition", None)
        if transition and transition.get("left_acquisition", False) and not hasattr(self.supervisor, "is_offset_latched"):
            self._offset_xy = self._offset_from_meas(self.last_y_meas)
            self._offset_latched_fallback = True

        # 3. system owns the swap — including warm-start on switches
        self._sync_active_components(ctrl_i, est_i, x_hat=x_hat)

        self.x = x_true
        self.u = u_cmd
        if hasattr(self.supervisor, "note_applied_command"):
            self.supervisor.note_applied_command(u_cmd)

        self.step_data = StepData(
            x=x_true,
            u=u_cmd,
            acc=acc,
            innovation=innovation,
            mech_joints=mech_joints,
            offset_xy=self._offset_xy.copy(),
            offset_latched=self._is_offset_latched(),
        )

    def reset(self):
        self.active_plant = self.plants[0]
        self.active_controller = self.controllers[0]
        self.active_estimator = self.estimators[0]

        if hasattr(self.supervisor, "attach_runtime"):
            self.supervisor.attach_runtime(actuator=self.actuator, workspace=self.workspace)
        if hasattr(self.supervisor, "reset"):
            self.supervisor.reset()
        self._sync_active_components(*self._supervisor_active_indices())
        self.active_controller.reset()
        self.active_estimator.reset()
        for plant in self.plants:
            if hasattr(plant, "reset"):
                plant.reset()
        if hasattr(self.sensor, "reset"):
            self.sensor.reset()
        self.x = self.random_state()
        self.u = ControlInput(px_cmd=0, py_cmd=0)
        if hasattr(self.supervisor, "note_applied_command"):
            self.supervisor.note_applied_command(self.u)
        self.last_y_meas = None
        self._offset_latched_fallback = False
        self._offset_xy = self._offset_from_state(self.x)
        self._reset_fall_detection()

        mj0 = self.actuator.mech_joint_snapshot(self.u)
        self.step_data = StepData(
            x=self.x,
            u=self.u,
            acc=TableAccel(x_ddot=0.0, y_ddot=0.0),
            innovation=np.zeros(4),
            mech_joints=mj0,
            offset_xy=self._offset_xy.copy(),
            offset_latched=self._is_offset_latched(),
        )
    
    
    # doesn't take x_ref into account yet
    def random_state(self) -> State:
        spread  = lambda r: np.random.uniform(-r, r)
        
        angle = np.deg2rad(self.init_spread.ang_deg)
        pos = self.init_spread.pos_m
        v_spread = self.init_spread.vel_mps
        w_spread = np.deg2rad(self.init_spread.w_degps)

        return State(
            px = spread(pos),
            vx = spread(v_spread),
            ax = spread(angle),
            wx = spread(w_spread),

            py = spread(pos),
            vy = spread(v_spread),
            ay = spread(angle),
            wy = spread(w_spread),
        )
