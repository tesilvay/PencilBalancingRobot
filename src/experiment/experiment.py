from dataclasses import dataclass

EXPERIMENT_PRESETS = {
    "sim": {
        "system":          "default:dynamic_sim",
        "logger":          "default:default",
        "stop_condition":  "max_steps:default",
        "visualizer":      {"animation": "animation:default"},
        "progress":        "progress_bar:default",
    },
    "real": {
        "base": "sim", 
        "system": "default:real",
        "visualizer": {"realtime":"realtime:default"}
    },
    "headless": {"base": "sim", "visualizer": "none:default"},
}

@dataclass
class ExperimentParams:
    system:         object                  # runs step
    logger:         object                  # logs data
    stop_condition: object           # determines when to stop simulation
    visualizers:     dict   # does realtime render or post animation
    progress:       object                # shows progress bar of simulation, or simulations
    pacing:         object                  # determines realtime or offline pacing?
    scheduler:      object               # when to actuate (if dt and actuator dt doesnt match) or when to render
    
    


class Experiment:
    def __init__(self, params:ExperimentParams):
        p=params
        
        self.system = p.system
        self.logger = p.logger
        self.stop_condition = p.stop_condition
        self.visualizers = p.visualizers
        self.progress = p.progress
        self.pacing = p.pacing
        self.scheduler = p.scheduler
        
    def run_trial(self):
        self.system.reset()
        while not self.stop_condition.should_stop(self.system.state):
            self.system.step()
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