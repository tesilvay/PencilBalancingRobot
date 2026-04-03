class Experiment:
    def __init__(self, params):
        p = params

        self.system         = p.system
        self.logger         = p.logger
        self.stop_condition = p.stop_condition
        self.visualizers    = p.visualizer   # preset key is "visualizer" (singular)
        self.progress       = p.progress
        self.pacing         = p.pacing
        self.scheduler      = p.scheduler

    def run(self):
        return None
