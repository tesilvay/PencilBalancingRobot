from dataclasses import dataclass, field

from src.shared import TimingParams, default_timing


EXPERIMENT_PRESETS = {
    "sim": {
        "system":              "default:simple_sim",
        "logger":              "default:default",
        "stop_condition":      "max_steps:default",
        "realtime_visualizer": "null:default",
        "offline_visualizer":  "3d:default",
        "progress":            "default:default",
        "pacing":              "null:default",
        "scheduler":           "realtime:default",
        "n_trials":            1,
    },
    "realtime_sim": {
        "base": "sim",
        "realtime_visualizer": "sim:default",
        "offline_visualizer":  "null:default",
        "pacing":              "realtime:default",
    },
    "real": {
        "base": "sim",
        "system":              "default:real",
        "realtime_visualizer": "real:default",
        "offline_visualizer":  "null:default",
        "pacing":              "realtime:default",
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
                )
                if vr.quit:
                    break
            self.pacing.pace()
            i += 1

        result = self.logger.get_result()
        self.offline_visualizer.finalize(result, dt=self.dt)
        return result

    def reset(self):
        self.system.reset()
        self.logger.reset(self.system.x, self.system.u)
        self.stop_condition.reset()
        self.scheduler.reset()

    def run_experiment(self):
        results = []

        for _ in range(self.n_trials):
            result = self.run_trial()
            results.append(result)

        return results
