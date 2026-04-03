from new_architecture.params import ExperimentParams


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