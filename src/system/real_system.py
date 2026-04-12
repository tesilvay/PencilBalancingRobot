from dataclasses import dataclass, field
import time

import numpy as np
from .system import System
from src.system.actuator.servo_workspace_offset_calibrator import calibrate_servo_workspace_offset
from src.shared import (
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
class RealSystemParams:
    plants:       list
    controllers: list
    estimators:  list
    sensor:      object
    actuator:    object
    supervisor:  object
    gain_schedule: object
    init_spread: InitConditionsSpread = field(default_factory=default_spread)
    workspace:   WorkspaceParams     = field(default_factory=default_workspace)
    fall_angle_deg: float            = 20.0
    fall_hold_s: float               = 0.03




REAL_SYSTEM_PRESETS = {
    "real_vision": {
        "plants":       ["sim:default"],
        "controllers": ["smooth_pole:smoother"],
        "estimators":  ["lpf:test"],
        "sensor":      "real_dvs:hough",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
        "gain_schedule": "null:default",
    },
    "real": {
        "base": "real_vision",
        "actuator": "servo:default",
        "gain_schedule": "null:default",
    },
    "real_supervised_best": {
        "plants": ["sim:default", "sim:default"],
        "sensor":      "real_dvs:hough",
        "controllers": ["null:default", "smooth_pole:test1"],
        "estimators":  ["lpf:default", "kalman:test1"],
        "actuator": "servo:default",
        "supervisor": "real:default",
        
        "gain_schedule": "null:default", #or power
    },
    "real_supervised": {
        "base": "real_supervised_best",
        "controllers": ["null:default", "smooth_pole:lead"],
        "estimators":  ["lpf:lead"],
    },
    "real_cmd_state": {
        "base": "real_supervised_best",
        "controllers": ["null:default", "smooth_pole_cmd_state:default"],
    },
    "real_dynamic_supervised": {
        "base": "real_supervised",
        "estimators":  ["lpf:test2", "kalman:test1"],
        "supervisor": "real_dynamic:default",
        "gain_schedule": "null:default",
    },
    "real_new_sim": {
        "base": "real_supervised_best",
        "estimators":  ["lpf:lead"],
        "plants": ["accel_sim:default", "accel_sim:default"],
        "controllers": ["null:default", "accel_pole:default"],
    },
}






class RealSystem(System):
    @property
    def is_simulation(self) -> bool:
        return False

    def step(self, dt, control_tick: bool = True):
        prev_prestart = bool(getattr(self.supervisor, "is_prestart_state", False))
        ctrl_i, est_k = self._supervisor_active_output()
        self._sync_active_components(ctrl_i, est_k)

        # Preserve the real DVS startup fallback path: before both trackers lock,
        # the sensor can synthesize a measurement from the internal state.
        y_raw = self.sensor.get_y(self.x)
        y_shaped = self.gain_schedule.apply(y_raw)
        self.last_y_raw = y_shaped

        if not self._is_offset_latched():
            self._offset_xy = self._offset_from_meas(y_shaped)
        y = self._measurement_with_offset(
            y_shaped,
            self._offset_xy,
            self._angle_offset_from_supervisor(),
        )
        self.last_y_meas = y

        self.last_estimates, self.last_innovations = self._run_estimators(y, dt)
        x_used = self._blend_state_estimates(est_k)
        innovation_used = self._blend_innovations(est_k)
        adaptive_lpf_weight_used = self._blend_adaptive_lpf_weight(est_k)

        self._print_estimator_estimates(est_k)

        u_cmd = self._compute_command(x_used) if control_tick else self.u
        mech_joints = self._apply_or_hold_command(u_cmd, control_tick)
        x_true, acc = self.active_plant.step(self.x, u_cmd, dt)
        if self._update_fall_detection(x_used, dt):
            self.supervisor.notify_fall_detected()

        x_hat_0 = self.last_estimates[0] if self.last_estimates else x_used
        innovation_0 = self.last_innovations[0] if self.last_innovations else innovation_used
        x_hat_1 = self.last_estimates[1] if len(self.last_estimates) > 1 else x_hat_0
        innovation_1 = self.last_innovations[1] if len(self.last_innovations) > 1 else innovation_0
        ctrl_i, est_k = self.supervisor.update(x_hat_0, innovation_0, x_hat_1, innovation_1, dt)
        transition = getattr(self.supervisor, "last_transition", None)
        x_reset = x_used
        if self._left_acquisition(prev_prestart):
            x_reset = self._state_from_measurement(y)
            self._reset_fall_detection()
            self._reset_estimators(x_reset)
        if transition and transition.get("left_prestart", False) and not hasattr(self.supervisor, "is_offset_latched"):
            source_y = self.last_y_raw if self.last_y_raw is not None else self.last_y_meas
            self._offset_xy = self._offset_from_meas(source_y)
            self._offset_latched_fallback = True

        self._sync_active_components(ctrl_i, est_k, x_hat=x_reset)

        self.x = x_true
        self.u = u_cmd
        if hasattr(self.supervisor, "note_applied_command"):
            self.supervisor.note_applied_command(u_cmd)

        self.step_data = StepData(
            x=x_true,
            u=u_cmd,
            acc=acc,
            x_hat=x_used,
            innovation=innovation_used,
            mech_joints=mech_joints,
            offset_xy=self._offset_xy.copy(),
            offset_latched=self._is_offset_latched(),
            supervisor_state=getattr(self.supervisor, "state_name", None),
            adaptive_lpf_weight=adaptive_lpf_weight_used,
        )

    def reset(self):
        self._maybe_run_startup_calibration()
        self.active_plant = self.plants[0]
        self.active_controller = self.controllers[0]
        self.active_estimator = self.estimators[0]

        if hasattr(self.supervisor, "attach_runtime"):
            self.supervisor.attach_runtime(actuator=self.actuator, workspace=self.workspace)
        if hasattr(self.supervisor, "reset"):
            self.supervisor.reset()
        self._sync_active_components(*self._supervisor_active_output())
        self.active_controller.reset()
        for estimator in self.estimators:
            estimator.reset()
        for plant in self.plants:
            if hasattr(plant, "reset"):
                plant.reset()
        if hasattr(self.sensor, "reset"):
            self.sensor.reset()

        self.x = State(
            px=0.0, vx=0.0, ax=0.0, wx=0.0,
            py=0.0, vy=0.0, ay=0.0, wy=0.0,
        )
        self.u = ControlInput(px_cmd=float(self.workspace.x_ref), py_cmd=float(self.workspace.y_ref))
        if hasattr(self.supervisor, "note_applied_command"):
            self.supervisor.note_applied_command(self.u)
        self.last_y_meas = None
        self.last_y_raw = None
        self.last_estimates = []
        self.last_innovations = []
        self._last_estimator_print_t = 0.0
        self._offset_latched_fallback = False
        self._offset_xy = np.zeros(2, dtype=float)
        self._reset_fall_detection()

        mj0 = self.actuator.mech_joint_snapshot(self.u)
        self.step_data = StepData(
            x=self.x,
            u=self.u,
            acc=TableAccel(x_ddot=0.0, y_ddot=0.0),
            x_hat=self.x,
            innovation=np.zeros(4),
            mech_joints=mj0,
            offset_xy=self._offset_xy.copy(),
            offset_latched=self._is_offset_latched(),
            supervisor_state=getattr(self.supervisor, "state_name", None),
            adaptive_lpf_weight=self._blend_adaptive_lpf_weight(self.active_est_k),
        )
