class ExperimentRunner:
    def __init__(
        self,
        system,
        scheduler,
        stop_condition,
        pacing,
        logger=None,
        actuator=None,
        visualizer=None,
    ):
        self.system = system
        self.scheduler = scheduler
        self.stop_condition = stop_condition
        self.pacing = pacing

        self.logger = logger
        self.actuator = actuator
        self.visualizer = visualizer

        self.command = None
        self.state = None

    def initialize(self, initial_state, initial_command):
        self.state = initial_state
        self.command = initial_command

        if self.logger:
            self.logger.reset(initial_state, initial_command)
        

    def run(self):
        i = 0

        while not self.stop_condition.should_stop(i, self.state, self.scheduler.dt):

            # ---- 1. advance system ----
            (
                state_true,
                state_est,
                self.command,
                acc,
                measurement,
                pose,
            ) = self.system.step(self.state, self.command)

            self.state = state_true

            # ---- 2. actuator ----
            if self.actuator and self.scheduler.should_actuate():
                self.actuator.send(self.command)

            # ---- 3. visualization ----
            if self.visualizer and self.scheduler.should_render():
                self.visualizer.render(
                    measurement=measurement,
                    command=self.command,
                    pose=pose,
                )

            # ---- 4. logging ----
            if self.logger:
                self.logger.record(
                    state=self.state,
                    command=self.command,
                    acc=acc,
                )

            # ---- 5. time update ----
            self.scheduler.tick()
            self.pacing.pace()

            i += 1
    
        if self.logger:
            return self.logger.get_result()
