from dataclasses import dataclass, field

import numpy as np

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
class SystemParams:
    plants:       list
    # Ordered lists of collaborators sharing a common base type. Supervisors
    # select active instances by integer index; order must match preset + supervisor.
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




SYSTEM_PRESETS = {
    "simple_sim": {
        "plants":       ["sim:default"],
        "controllers": ["smooth_pole:default"],
        "estimators":  ["lpf:test"],
        "sensor":      "sim_analytic:noisy",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
        "gain_schedule": "null:default",
    },
    "placing_only": {
        "plants":       ["sim:default"],#["placing:angle_only"],
        "controllers": ["smooth_pole:default"],
        "estimators":  ["kalman:default"],
        "sensor":      "sim_dvs:hough",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
        "gain_schedule": "null:default",
    },
    "dynamic_sim": {
        "plants":       ["placing:steady_hands", "sim:default"],
        "controllers": ["smooth_pole:default"],
        "estimators":  ["lpf:default", "lpf:default"],
        "sensor":      "sim_analytic:noisy",
        "actuator":    "mock:default",
        "supervisor":  "dynamic:default",
        "gain_schedule": "null:default",
    },
    "new_sim": {
        "base": "simple_sim",
        "estimators":  ["lpf:lead"],
        "sensor":      "sim_analytic:noisy",
        "plants": ["accel_sim:default"],
        "controllers": ["accel_pole:default"],
        "gain_schedule": "null:default",
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
        self.gain_schedule = params.gain_schedule
        
        #self.gain_schedule.plot_angle_shape()
        
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
        self.last_y_raw = None
        
        self.last_estimates: list[State] = []
        self.last_innovations: list[np.ndarray] = []
        
        self.i = 0
        
        # printing stuff hardcoded
        self.print_every_n_steps = 20
        
        self._offset_xy = np.zeros(2, dtype=float)
        self._offset_latched_fallback = False
        self.fall_angle_rad = float(np.deg2rad(params.fall_angle_deg))
        self.fall_hold_s = float(params.fall_hold_s)
        self._fall_detected = False
        self._fall_timer_s = 0.0
        self._startup_calibration_done = False
        self._perf_tip_history: list[np.ndarray] = []
        self._perf_tip_ref_history: list[np.ndarray] = []
        self._perf_orientation_history: list[np.ndarray] = []
        self._perf_control_history: list[np.ndarray] = []
        self._perf_time_history: list[float] = []
        self._perf_filtered_tip_history: list[np.ndarray] = []
        self._perf_lp_tip: np.ndarray | None = None
        self._perf_noise_tau_s = 0.05
        self._perf_settling_eps_m = 2e-3
        if hasattr(self.supervisor, "attach_runtime"):
            self.supervisor.attach_runtime(actuator=self.actuator, workspace=self.workspace)

    @property
    def is_simulation(self) -> bool:
        return True

    def _maybe_run_startup_calibration(self) -> None:
        if self._startup_calibration_done:
            return

        mechanism = getattr(self.actuator, "mechanism", None)
        serial_handle = getattr(self.actuator, "_serial", None)
        if mechanism is None or serial_handle is None:
            self._startup_calibration_done = True
            return

        calibrate_servo_workspace_offset(
            system=self,
            actuator=self.actuator,
            workspace=self.workspace,
        )
        self._startup_calibration_done = True

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

    def _angle_offset_from_supervisor(self) -> np.ndarray:
        angle_offset = getattr(self.supervisor, "measurement_angle_offset", (0.0, 0.0))
        return np.asarray(angle_offset, dtype=float).reshape(2)

    def _measurement_with_offset(
        self,
        y: Measurement,
        offset_xy: np.ndarray,
        angle_offset: np.ndarray,
    ) -> Measurement:
        return Measurement(
            px=float(y.px - offset_xy[0]),
            py=float(y.py - offset_xy[1]),
            ax=float(y.ax - angle_offset[0]),
            ay=float(y.ay - angle_offset[1]),
        )

    def finalize_command(self, u_raw: ControlInput) -> ControlInput:
        u_applied = clamp_control_input_to_workspace(u_raw, self.workspace)
        self.active_controller.set_applied_command(u_applied)
        return u_applied

    def _compute_command(self, x_used: State) -> ControlInput:
        u_raw = self.active_controller.compute(x_used)
        u_override = getattr(self.supervisor, "command_override", None)
        if u_override is not None:
            u_raw = u_override
        return self.finalize_command(u_raw)

    def _apply_or_hold_command(self, u_cmd: ControlInput, control_tick: bool) -> np.ndarray:
        if control_tick:
            return self.actuator.apply(u_cmd)
        return self.actuator.mech_joint_snapshot(u_cmd)

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
        if not self.controllers:
            raise RuntimeError("System requires at least one controller.")
        if not self.plants:
            raise RuntimeError("System requires at least one plant.")

        if len(self.controllers) == 1:
            ctrl_i = 0
        new_controller = self.controllers[ctrl_i]
        if new_controller is not self.active_controller:
            new_controller.reset(x_hat)

        plant_i = min(self._dominant_estimator_index(est_k), len(self.plants) - 1)
        self.active_plant = self.plants[plant_i]
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

    @staticmethod
    def _estimator_adaptive_lpf_weight(estimator: object) -> float | None:
        value = getattr(estimator, "adaptive_lpf_weight", None)
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return float(np.clip(value, 0.0, 1.0))

    def _blend_adaptive_lpf_weight(self, est_k: float) -> float | None:
        if not self.estimators:
            return None

        weights = [
            self._estimator_adaptive_lpf_weight(estimator)
            for estimator in self.estimators[:2]
        ]
        if not any(weight is not None for weight in weights):
            return None
        if len(weights) == 1:
            return 0.0 if weights[0] is None else weights[0]

        k = float(np.clip(est_k, 0.0, 1.0))
        w0 = 0.0 if weights[0] is None else weights[0]
        w1 = 0.0 if weights[1] is None else weights[1]
        return float((1.0 - k) * w0 + k * w1)

    def _print_estimator_estimates(self, est_k: float) -> None:
        if self.i % self.print_every_n_steps != 0:
            return

        if len(self.estimators) == 2 and len(self.last_estimates) == 2:
            k = float(np.clip(est_k, 0.0, 1.0))
            est1_weight = 1.0 - k
            est2_weight = k
            print(f"est 1 [{est1_weight:.2f}] : {self.last_estimates[0].state_str()}")
            print(f"est 2 [{est2_weight:.2f}] : {self.last_estimates[1].state_str()}")
            return

        for idx, (estimator, x_hat) in enumerate(zip(self.estimators, self.last_estimates)):
            print(f"est {idx + 1}: {x_hat.state_str()}")
    
    def _print_x_hat_and_true(self, x_true: State, dt: float) -> None:
        if self.i % self.print_every_n_steps != 0:
            return

        sim_time = self.i * dt
        print(f"time: {sim_time:.3f}")
        print(f"est : {self.last_estimates[0].state_str()}")
        print(f"true: {x_true.state_str()}")

    def _performance_com_length(self) -> float:
        for owner in (self.active_plant, self.active_controller, self.active_estimator):
            length = getattr(owner, "l", None)
            if length is not None:
                return float(length)
            plant = getattr(owner, "_plant", None)
            if plant is not None and hasattr(plant, "com_length"):
                return float(plant.com_length)
        return 0.0

    def _reference_measurement(self) -> Measurement:
        return Measurement(
            px=float(self.workspace.x_ref),
            py=float(self.workspace.y_ref),
            ax=0.0,
            ay=0.0,
        )

    def _tip_from_measurement(self, y: Measurement) -> np.ndarray:
        l = self._performance_com_length()
        return np.array(
            [
                float(y.px + l * y.ax),
                float(y.py + l * y.ay),
                l,
            ],
            dtype=float,
        )

    def _orientation_from_measurement(self, y: Measurement) -> np.ndarray:
        n = np.array(
            [
                np.sin(float(y.ax)),
                np.sin(float(y.ay)),
                np.cos(float(y.ax)) * np.cos(float(y.ay)),
            ],
            dtype=float,
        )
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return n / n_norm

    def _update_performance_history(self, y_meas: Measurement, u_cmd: ControlInput, dt: float) -> None:
        tip = self._tip_from_measurement(y_meas)
        tip_ref = self._tip_from_measurement(self._reference_measurement())
        orientation = self._orientation_from_measurement(y_meas)
        control = np.array([float(u_cmd.px_cmd), float(u_cmd.py_cmd)], dtype=float)
        sim_time = self.i * dt

        alpha = float(dt / (self._perf_noise_tau_s + dt)) if dt > 0.0 else 1.0
        if self._perf_lp_tip is None:
            self._perf_lp_tip = tip.copy()
        else:
            self._perf_lp_tip = (1.0 - alpha) * self._perf_lp_tip + alpha * tip

        self._perf_tip_history.append(tip)
        self._perf_tip_ref_history.append(tip_ref)
        self._perf_orientation_history.append(orientation)
        self._perf_control_history.append(control)
        self._perf_time_history.append(float(sim_time))
        self._perf_filtered_tip_history.append(self._perf_lp_tip.copy())

    def _compute_delay_indicator(self, dt: float) -> float:
        if len(self._perf_tip_history) < 2:
            return float("nan")

        y_hist = np.asarray(self._perf_tip_history, dtype=float)
        y_ref_hist = np.asarray(self._perf_tip_ref_history, dtype=float)
        y_centered = y_hist - np.mean(y_hist, axis=0, keepdims=True)
        y_ref_centered = y_ref_hist - np.mean(y_ref_hist, axis=0, keepdims=True)
        if float(np.linalg.norm(y_ref_centered)) <= 1e-12:
            return float("nan")

        corr = None
        for axis in range(y_centered.shape[1]):
            axis_corr = np.correlate(y_centered[:, axis], y_ref_centered[:, axis], mode="full")
            corr = axis_corr if corr is None else corr + axis_corr

        lag_samples = int(np.argmax(corr) - (len(y_hist) - 1))
        return float(lag_samples * dt)

    def _compute_settling_time(self) -> float:
        if not self._perf_time_history:
            return float("nan")

        error_norms = np.linalg.norm(
            np.asarray(self._perf_tip_history, dtype=float)
            - np.asarray(self._perf_tip_ref_history, dtype=float),
            axis=1,
        )
        below = error_norms < self._perf_settling_eps_m
        for idx in range(len(below)):
            if np.all(below[idx:]):
                return float(self._perf_time_history[idx])
        return float("nan")

    def _print_performance_indicators(self, dt: float) -> None:
        if self.i % self.print_every_n_steps != 0:
            return

        if not self._perf_tip_history:
            return

        tip_hist = np.asarray(self._perf_tip_history, dtype=float)
        tip_ref_hist = np.asarray(self._perf_tip_ref_history, dtype=float)
        orientation_hist = np.asarray(self._perf_orientation_history, dtype=float)
        control_hist = np.asarray(self._perf_control_history, dtype=float)
        filtered_tip_hist = np.asarray(self._perf_filtered_tip_history, dtype=float)
        err_hist = tip_hist - tip_ref_hist
        err_norms = np.linalg.norm(err_hist, axis=1)
        n_d = np.array([0.0, 0.0, 1.0], dtype=float)

        if len(tip_hist) >= 2 and dt > 0.0:
            vel_hist = np.diff(tip_hist, axis=0) / dt
            vel_ref_hist = np.diff(tip_ref_hist, axis=0) / dt
            control_rate_hist = np.diff(control_hist, axis=0) / dt
            j5 = float(dt * np.sum(np.sum((vel_hist - vel_ref_hist) ** 2, axis=1)))
            j7 = float(dt * np.sum(np.sum(control_rate_hist ** 2, axis=1)))
        else:
            j5 = 0.0
            j7 = 0.0

        j1 = self._compute_delay_indicator(dt)
        j2 = float(np.sqrt(np.mean(np.sum(err_hist ** 2, axis=1))))
        j3 = float(np.max(err_norms))
        j4 = float(dt * np.sum(np.sum(np.cross(orientation_hist, n_d) ** 2, axis=1)))
        j6 = float(dt * np.sum(np.sum(control_hist ** 2, axis=1)))
        j8 = self._compute_settling_time()
        noise_hist = tip_hist - filtered_tip_hist
        j9 = float(np.mean(np.sum(noise_hist ** 2, axis=1)))

        def _fmt(value: float, unit: str = "") -> str:
            if np.isnan(value):
                return f"{'N/A':>12}"
            return f"{value:>12.6f}{unit}"

        print("performance:")
        print("+------+--------------------------+----------------+")
        print("| ID   | indicador                | valor          |")
        print("+------+--------------------------+----------------+")
        print(f"| J1   | desfase temporal         | {_fmt(j1, ' s')} |")
        print(f"| J2   | error RMS                | {_fmt(j2, ' m')} |")
        print(f"| J3   | error pico               | {_fmt(j3, ' m')} |")
        print(f"| J4   | error geometrico         | {_fmt(j4)} |")
        print(f"| J5   | error velocidad          | {_fmt(j5)} |")
        print(f"| J6   | esfuerzo control         | {_fmt(j6)} |")
        print(f"| J7   | variacion control        | {_fmt(j7)} |")
        print(f"| J8   | tiempo establecimiento   | {_fmt(j8, ' s')} |")
        print(f"| J9   | indicador ruido          | {_fmt(j9)} |")
        print("+------+--------------------------+----------------+")

            

    def _left_acquisition(self, prev_prestart: bool) -> bool:
        transition = getattr(self.supervisor, "last_transition", None)
        if transition is not None:
            return bool(transition.get("left_acquisition", transition.get("left_prestart", False)))
        return prev_prestart and not bool(getattr(self.supervisor, "is_prestart_state", False))

    def _reset_estimators(self, x_hat: State | None = None) -> None:
        for estimator in self.estimators:
            estimator.reset(x_hat)

    @staticmethod
    def _state_from_measurement(y: Measurement) -> State:
        return State(
            px=float(y.px),
            vx=0.0,
            ax=float(y.ax),
            wx=0.0,
            py=float(y.py),
            vy=0.0,
            ay=float(y.ay),
            wy=0.0,
        )

    def step(self, dt, control_tick: bool = True):
        prev_prestart = bool(getattr(self.supervisor, "is_prestart_state", False))
        ctrl_i, est_k = self._supervisor_active_output()
        self._sync_active_components(ctrl_i, est_k)

        x_true, acc = self.active_plant.step(self.x, self.u, dt)

        # get measurements
        y_raw = self.sensor.get_y(x_true)
        y_shaped = self.gain_schedule.apply(y_raw)
        self.last_y_raw = y_shaped
        
        # no latch, keep updating offset
        if not self._is_offset_latched():
            self._offset_xy = self._offset_from_meas(y_shaped)
        y = self._measurement_with_offset(
            y_shaped,
            self._offset_xy,
            self._angle_offset_from_supervisor(),
        )
        self.last_y_meas = y

        # Keep every estimator warm, then build the controller-facing blended estimate.
        self.last_estimates, self.last_innovations = self._run_estimators(y, dt)
        x_used = self._blend_state_estimates(est_k)
        innovation_used = self._blend_innovations(est_k)
        adaptive_lpf_weight_used = self._blend_adaptive_lpf_weight(est_k)

        # self._print_estimator_estimates(est_k)
        # self._print_x_hat_and_true(x_true, dt)

        u_cmd = self._compute_command(x_used) if control_tick else self.u
        mech_joints = self._apply_or_hold_command(u_cmd, control_tick)
        #self._update_performance_history(y, u_cmd, dt)
        #self._print_performance_indicators(dt)
        if self._update_fall_detection(x_used, dt):
            self.supervisor.notify_fall_detected()

        # 2. supervisor decides next controller + estimator blend
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

        # 3. system owns the controller swap while estimator usage is blended.
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
        self.i += 1

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
        if hasattr(self.gain_schedule, "reset"):
            self.gain_schedule.reset()
        self.x = self.random_state()
        self.u = ControlInput(px_cmd=0, py_cmd=0)
        if hasattr(self.supervisor, "note_applied_command"):
            self.supervisor.note_applied_command(self.u)
        self.last_y_meas = None
        self.last_y_raw = None
        self.last_estimates = []
        self.last_innovations = []
        self.i = 0
        self._offset_latched_fallback = False
        self._offset_xy = self._offset_from_state(self.x)
        self._reset_fall_detection()
        self._perf_tip_history = []
        self._perf_tip_ref_history = []
        self._perf_orientation_history = []
        self._perf_control_history = []
        self._perf_time_history = []
        self._perf_filtered_tip_history = []
        self._perf_lp_tip = None

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
