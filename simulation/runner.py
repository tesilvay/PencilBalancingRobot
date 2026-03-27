from core.sim_types import SimulationResult, TerminalInfo
import cv2

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
                surfaces = None
                try:
                    vision = getattr(self.system, "perception", None)
                    vision = getattr(vision, "vision", None)
                    if vision is not None and hasattr(vision, "get_surfaces"):
                        surfaces = vision.get_surfaces()
                except Exception:
                    surfaces = None
                viz_result = self.visualizer.render(
                    measurement=measurement,
                    command=self.command,
                    pose=pose,
                    surfaces=surfaces,
                )
                quit_requested = False
                if isinstance(viz_result, tuple):
                    # DVS visualizer returns (quit_requested, toggle_pause)
                    quit_requested = bool(viz_result[0])
                elif isinstance(viz_result, bool):
                    quit_requested = viz_result
                if quit_requested:
                    break

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
    
        terminal = TerminalInfo(
            stabilized=self.stop_condition.is_stabilized(),
            settling_time=self.stop_condition.settling_time()
        )
        result = self.logger.get_result()
        cv2.destroyAllWindows()
        
        return SimulationResult(
            state_history=result.state_history,
            acc_history=result.acc_history,
            cmd_history=result.cmd_history,
            terminal=terminal
        )
