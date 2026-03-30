from core.sim_types import SimulationResult, TableCommand, TerminalInfo, clamp_table_command_to_workspace
from visualization.realtime_visualizer import VizResult
import cv2




class ExperimentRunner:
    def __init__(
        self,
        system,
        controller,
        scheduler,
        stop_condition,
        pacing,
        workspace,
        logger=None,
        actuator=None,
        visualizer=None,
    ):
        self.system = system
        self.controller = controller
        self.scheduler = scheduler
        self.stop_condition = stop_condition
        self.pacing = pacing

        self.logger = logger
        self.actuator = actuator
        self.visualizer = visualizer
        self.workspace = workspace

        self.command = None
        self.state = None
        self._viz_paused = False

    def _reset_conditions(self):
        self.controller.reset()
        perception = getattr(self.system, "perception", None)
        if perception is not None and hasattr(perception, "reset"):
            perception.reset()
        self.scheduler.reset()
        if self.stop_condition is not None and hasattr(
            self.stop_condition, "reset"
        ):
            self.stop_condition.reset()
        if self.actuator is not None and hasattr(self.actuator, "reset"):
            self.actuator.reset()
        if self.visualizer is not None and hasattr(self.visualizer, "reset"):
            self.visualizer.reset()

    def initialize(self, initial_state, initial_command):
        self._reset_conditions()
        self.state = initial_state
        self.command = initial_command
        self._viz_paused = False

        if self.logger:
            self.logger.reset(initial_state, initial_command)

    def _compute_command(self, state_est):
        u_raw = self.controller.compute(state_est)
        command = clamp_table_command_to_workspace(u_raw, self.workspace)
        if hasattr(self.controller, "set_applied_command"):
            self.controller.set_applied_command(command)
        return command

    def _workspace_center_command(self) -> TableCommand:
        ws = self.workspace
        return clamp_table_command_to_workspace(TableCommand(ws.x_ref, ws.y_ref), ws)


    def run(self):
        i = 0

        while not self.stop_condition.should_stop(i, self.state, self.scheduler.dt):

            # ---- 1. advance system ----
            (
                state_true,
                state_est,
                acc,
                measurement,
                pose,
            ) = self.system.step(self.state, self.command)

            self.state = state_true

            # ---- 2. actuator ----
            # Paused UI means "table at center": drive real servos there, not the live controller output.
            if self.scheduler.should_actuate():
                
                self.command = self._compute_command(state_est)
                
                cmd_out = (
                    self._workspace_center_command()
                    if self._viz_paused
                    else self.command
                )
                self.actuator.send(cmd_out)

            # ---- 3. visualization ----
            if self.visualizer and self.scheduler.should_render():
                viz_result = self.visualizer.render(
                    measurement=measurement,
                    command=self.command,
                    pose=pose,
                    paused=self._viz_paused,
                )
                if isinstance(viz_result, VizResult):
                    if viz_result.toggle_pause:
                        self._viz_paused = not self._viz_paused
                        # Same frame as keypress we may have already sent the controller command above.
                        if self._viz_paused and self.actuator:
                            self.actuator.send(self._workspace_center_command())
                    if viz_result.quit:
                        break
                elif isinstance(viz_result, tuple):
                    if bool(viz_result[0]):
                        break
                elif viz_result:
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
