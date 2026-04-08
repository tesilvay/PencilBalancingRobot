from dataclasses import dataclass, field

from src.experiment.logger import TerminalInfo
from src.shared import TimingParams, default_timing


EXPERIMENT_PRESETS = {
    "sim": {
        "system":              "default:simple_sim",
        "logger":              "default:default",
        "stop_condition":      "max_steps:default",
        "realtime_visualizer": "sim_ws:default",
        "offline_visualizer":  "3d:default",
        "progress":            "default:default",
        "pacing":              "null:default",
        "scheduler":           "realtime:default",
        "n_trials":            1,
    },
    "new_sim": {
        "base": "sim",
        "system":"default:new_sim",
        "stop_condition":  "max_steps:default",
        "realtime_visualizer": "real_ws:default",
        "offline_visualizer":  "3d:default",
    },
    "montecarlo": {
        "base": "sim",
        "system":"default:new_sim",
        "stop_condition":  "any:early_stop",
        "realtime_visualizer": "null:default",
        "offline_visualizer":  "null:default",
        "n_trials":            10,
    },
    "test_sim_dvs": {
        "base": "sim",
        "system":"default:placing_only",
        "realtime_visualizer": "real_ws:default",
        "offline_visualizer":  "3d:default",
    },
    "placing": {
        "base": "sim",
        "system":"default:placing_only",
    },
    "dynamic_sim": {
        "base": "sim",
        "system":"default:dynamic_sim",
        "offline_visualizer":  "3d:default",
    },
    "realtime_sim": {
        "base": "sim",
        "realtime_visualizer": "sim:default",
        "offline_visualizer":  "null:default",
        "pacing":              "realtime:default",
    },
    "real_vision": {
        "base": "sim",
        "system":              "default:real_vision",
        "stop_condition":      "infinite:default",
        "realtime_visualizer": "real_ws:default",
        "offline_visualizer":  "null:default",
        "pacing":              "realtime:default",
    },
    "real": {
        "base":   "real_vision",
        "system": "default:real",
    },
    "real_supervised": {
        "base":   "real_vision",
        "system": "default:real_supervised",
    },
    "real_supervised_dynamic": {
        "base":   "real_vision",
        "system": "default:real_dynamic_supervised",
    },
}


@dataclass
class ExperimentParams:
    system:               object
    logger:               object
    stop_condition:       object
    realtime_visualizer:  object
    offline_visualizer:   object
    progress:             object
    pacing:               object
    scheduler:            object
    n_trials:             int
    timing:               TimingParams = field(default_factory=default_timing)


class Experiment:
    def __init__(self, params: ExperimentParams):
        p = params

        self.system              = p.system
        self.logger               = p.logger
        self.stop_condition       = p.stop_condition
        self.realtime_visualizer  = p.realtime_visualizer
        self.offline_visualizer   = p.offline_visualizer
        self.progress             = p.progress
        self.pacing               = p.pacing
        self.scheduler            = p.scheduler
        self.dt                   = p.timing.dt
        self.n_trials             = p.n_trials

        if hasattr(self.realtime_visualizer, "_event_frames_fn"):
            if self.realtime_visualizer._event_frames_fn is None and hasattr(
                self.system.sensor, "get_event_accumulator_frames"
            ):
                self.realtime_visualizer._event_frames_fn = (
                    self.system.sensor.get_event_accumulator_frames
                )

    def _supervisor_title(self) -> str | None:
        state_name = getattr(self.system.supervisor, "state_name", None)
        if state_name == "SERVO_CENTERING":
            return "Centering | Arrows/WASD: move | Enter: accept | R: reset | Q: quit"
        if state_name == "ACQUISITION":
            return "Acquisition | Hold pencil upright to start motion | Space: reacquire | Q: quit"
        if state_name == "STABILIZING":
            return "Stabilizing | blending estimators | Space: reset | Q: quit"
        if state_name == "BALANCED":
            return "Balanced | Kalman fully blended | Space: reset | Q: quit"
        return None

    def run_trial(self):
        self.reset()
        i = 0
        while not self.stop_condition.should_stop(i, self.system.x, self.dt):
            self.system.step(self.dt)
            self.logger.record(self.system.step_data)
            self.scheduler.tick()
            if self.scheduler.should_render():
                meas = getattr(self.system.sensor, "last_line_observation", None)
                vr = self.realtime_visualizer.render(
                    measurement=meas,
                    command=self.system.u,
                    y_meas=self.system.last_y_meas,
                    paused=False,
                    title=self._supervisor_title(),
                )
                if vr.key is not None and hasattr(self.system.supervisor, "handle_key"):
                    self.system.supervisor.handle_key(vr.key)
                if vr.quit:
                    break
            self.pacing.pace()
            i += 1

        result = self.logger.get_result()
        result.terminal = TerminalInfo(
            stabilized=self.stop_condition.is_stabilized(),
            settling_time=self.stop_condition.settling_time(),
        )
        self.offline_visualizer.finalize(result, dt=self.dt)
        return result

    def reset(self):
        self.system.reset()
        self.logger.reset(self.system.step_data)
        self.stop_condition.reset()
        self.scheduler.reset()

    def run_experiment(self):
        results = []

        for _ in range(self.n_trials):
            result = self.run_trial()
            results.append(result)

        return results
