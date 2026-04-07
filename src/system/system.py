from dataclasses import dataclass, field
import time

import numpy as np

from src.system.estimator.kalman import KalmanEstimator
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
        "controllers": ["null:default", "smooth_pole:test1"],
        "estimators":  ["lpf:test2", "kalman:test"],
        "actuator": "servo:default",
        "supervisor": "real:default",
    },
    "real_dynamic_supervised": {
        "base": "real_supervised",
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
        self.active_est_k = 0.0
        
        
        self.step_data: StepData | None = None
        self.x = None
        self.u = None
        self.last_y_meas = None
        
        self.last_estimates: list[State] = []
        self.last_innovations: list[np.ndarray] = []
        
        # printing stuff hardcoded
        self.print_hz = 24.0
        self._last_estimator_print_t = 0.0
        
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

    def _supervisor_active_output(self) -> tuple[int, float]:
        try:
            active_output = self.supervisor.active_output
        except (AttributeError, NotImplementedError):
            ctrl_i = self.controllers.index(self.active_controller)
            return ctrl_i, self.active_est_k
        return active_output

    def _is_offset_latched(self) -> bool:
        if hasattr(self.supervisor, "is_offset_latched"):
            return bool(self.supervisor.is_offset_latched)
        return self._offset_latched_fallback

    def _dominant_estimator_index(self, est_k: float) -> int:
        if len(self.estimators) < 2:
            return 0
        return 0 if est_k < 0.5 else 1

    def _sync_active_components(self, ctrl_i: int, est_k: float, x_hat: State | None = None) -> None:
        new_controller = self.controllers[ctrl_i]
        if new_controller is not self.active_controller:
            new_controller.reset(x_hat)

        self.active_plant = self.plants[ctrl_i]
        self.active_controller = new_controller
        self.active_estimator = self.estimators[self._dominant_estimator_index(est_k)]
        self.active_est_k = float(np.clip(est_k, 0.0, 1.0))

    @property
    def fall_detected(self) -> bool:
        return self._fall_detected

    def _reset_fall_detection(self) -> None:
        self._fall_detected = False
        self._fall_timer_s = 0.0

    def _update_fall_detection(self, x_used: State, dt: float) -> bool:
        if getattr(self.supervisor, "is_prestart_state", False):
            self._reset_fall_detection()
            return False

        angle_norm = float(np.linalg.norm([x_used.ax, x_used.ay]))
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

    def _run_estimators(self, y: Measurement, dt: float) -> tuple[list[State], list[np.ndarray]]:
        estimates: list[State] = []
        innovations: list[np.ndarray] = []
        for estimator in self.estimators:
            x_hat, innovation = estimator.estimate(y_meas=y, dt=dt, u_cmd=self.u)
            estimates.append(x_hat)
            innovations.append(np.asarray(innovation, dtype=float).reshape(-1))
        return estimates, innovations

    def _blend_state_estimates(self, est_k: float) -> State:
        if not self.last_estimates:
            raise RuntimeError("No estimator states available to blend.")
        if len(self.last_estimates) == 1:
            return self.last_estimates[0]

        x0 = self.last_estimates[0].as_vector()
        x1 = self.last_estimates[1].as_vector()
        k = float(np.clip(est_k, 0.0, 1.0))
        return State.from_iterable((1.0 - k) * x0 + k * x1)

    def _blend_innovations(self, est_k: float) -> np.ndarray | None:
        if not self.last_innovations:
            return None
        if len(self.last_innovations) == 1:
            return self.last_innovations[0]

        k = float(np.clip(est_k, 0.0, 1.0))
        return (1.0 - k) * self.last_innovations[0] + k * self.last_innovations[1]

    def _print_estimator_estimates(self, est_k: float) -> None:
        min_period_s = 1.0 / self.print_hz
        now = time.perf_counter()
        elapsed = now - self._last_estimator_print_t
        if elapsed > min_period_s:
            self._last_estimator_print_t = time.perf_counter()

            if len(self.estimators) == 2 and len(self.last_estimates) == 2:
                k = float(np.clip(est_k, 0.0, 1.0))
                est1_weight = 1.0 - k
                est2_weight = k
                print(f"est 1 [{est1_weight:.2f}] : {self.last_estimates[0].state_str()}")
                print(f"est 2 [{est2_weight:.2f}] : {self.last_estimates[1].state_str()}")
                return

            for idx, (estimator, x_hat) in enumerate(zip(self.estimators, self.last_estimates)):
                print(f"est {idx + 1}: {x_hat.state_str()}")

    def _reset_kalman_estimators(self, x_hat: State | None = None) -> None:
        for estimator in self.estimators:
            if isinstance(estimator, KalmanEstimator):
                estimator.reset(x_hat)

    def step(self, dt):
        prev_supervisor_state = getattr(self.supervisor, "state_name", None)
        ctrl_i, est_k = self._supervisor_active_output()
        self._sync_active_components(ctrl_i, est_k)

        x_true, acc = self.active_plant.step(self.x, self.u, dt)

        # get measurements
        y = self.sensor.get_y(x_true)
        self.last_y_meas = y
        
        # no latch, keep updating offset
        if not self._is_offset_latched():
            self._offset_xy = self._offset_from_meas(y)
        y = self._measurement_with_offset(y, self._offset_xy)

        # Keep every estimator warm, then build the controller-facing blended estimate.
        self.last_estimates, self.last_innovations = self._run_estimators(y, dt)
        x_used = self._blend_state_estimates(est_k)
        innovation_used = self._blend_innovations(est_k)
        self._print_estimator_estimates(est_k)
        u_raw = self.active_controller.compute(x_used)
        u_override = getattr(self.supervisor, "command_override", None)
        if u_override is not None:
            u_raw = u_override
        u_cmd = self.finalize_command(u_raw)

        mech_joints = self.actuator.apply(u_cmd)
        if self._update_fall_detection(x_used, dt):
            self.supervisor.notify_fall_detected()

        # 2. supervisor decides next controller + estimator blend
        x_hat_0 = self.last_estimates[0] if self.last_estimates else x_used
        innovation_0 = self.last_innovations[0] if self.last_innovations else innovation_used
        x_hat_1 = self.last_estimates[1] if len(self.last_estimates) > 1 else x_hat_0
        innovation_1 = self.last_innovations[1] if len(self.last_innovations) > 1 else innovation_0
        ctrl_i, est_k = self.supervisor.update(x_hat_0, innovation_0, x_hat_1, innovation_1, dt)
        new_supervisor_state = getattr(self.supervisor, "state_name", None)
        if prev_supervisor_state == "ACQUISITION" and new_supervisor_state != "ACQUISITION":
            self._reset_fall_detection()
            self._reset_kalman_estimators(x_hat_0)
        transition = getattr(self.supervisor, "last_transition", None)
        if transition and transition.get("left_acquisition", False) and not hasattr(self.supervisor, "is_offset_latched"):
            self._offset_xy = self._offset_from_meas(self.last_y_meas)
            self._offset_latched_fallback = True

        # 3. system owns the controller swap while estimator usage is blended.
        self._sync_active_components(ctrl_i, est_k, x_hat=x_used)

        self.x = x_true
        self.u = u_cmd
        if hasattr(self.supervisor, "note_applied_command"):
            self.supervisor.note_applied_command(u_cmd)

        self.step_data = StepData(
            x=x_true,
            u=u_cmd,
            acc=acc,
            innovation=innovation_used,
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
        self._sync_active_components(*self._supervisor_active_output())
        self.active_controller.reset()
        for estimator in self.estimators:
            estimator.reset()
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
        self.last_estimates = []
        self.last_innovations = []
        self._last_estimator_print_t = 0.0
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
