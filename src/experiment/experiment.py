from dataclasses import dataclass, field
from src.shared import(
    TimingParams,
    default_timing
    
)


EXPERIMENT_PRESETS = {
    "sim": {
        "system":         "default:simple_sim",
        "logger":         "default:default",
        "stop_condition": "any:default",
        "visualizer":     ["3d:default"],
        "progress":       "default:default",
        "pacing":         "null:default",
        "scheduler":      "realtime:default",
    },
    "realtime_sim": {
        "base": "sim",
        "visualizer": ["sim:default"],
        "pacing":     "realtime:default",
    },
    "real": {
        "base": "sim",
        "system":     "default:real",
        "visualizer": ["real:default"],
        "pacing":     "realtime:default",
    },
}

@dataclass
class ExperimentParams:
    system:         object
    logger:         object
    stop_condition: object
    visualizer:     list
    progress:       object
    pacing:         object
    scheduler:      object
    timing:         TimingParams = field(default_factory=default_timing)
    



class Experiment:
    def __init__(self, params: ExperimentParams):
        p = params

        self.system         = p.system
        self.logger         = p.logger
        self.stop_condition = p.stop_condition
        self.visualizers    = p.visualizer   # preset key is "visualizer" (singular)
        self.progress       = p.progress
        self.pacing         = p.pacing
        self.scheduler      = p.scheduler
        self.dt             = p.timing.dt

    def run_trial(self):
        self.system.reset()
        while not self.stop_condition.should_stop(self.system.state):
            self.system.step(self.dt)
            self.logger.log(self.system.state)
            self.visualizer.update(self.system.state)  # real-time hook (no-op in sim)
        self.visualizer.render(self.logger.get_data())  # post-run hook (no-op in real)
        
        return self.logger.get_result()

    def run_experiment(self):
        results = []
        
        for _ in range(self.n_trials):
            result = self.run_trial()
            results.append(result)
        
        return results