import numpy as np
from core.system_builder import system_factory, runner_factory

from core.sim_types import (
    ExperimentSetup,
    SystemState,
    TableCommand,
    StopPolicy,
)


class SimulationEngine:
    def __init__(self, stop_policy: StopPolicy):
        self.stop_policy = stop_policy
    
    def run(self, setup: ExperimentSetup):
        system = system_factory(setup)
        runner = runner_factory(setup.params, system, self.stop_policy)

        initial_state, initial_command = self._initialize(setup)

        runner.initialize(initial_state, initial_command)
        return runner.run()

    def _initialize(self, setup):
        params = setup.params
        ws = params.workspace

        x_ref = SystemState(
            x=ws.x_ref, x_dot=0.0, alpha_x=0.0, alpha_x_dot=0.0,
            y=ws.y_ref, y_dot=0.0, alpha_y=0.0, alpha_y_dot=0.0
        )

        angle = np.deg2rad(params.run.initial_angle_spread_deg)
        spread = params.run.initial_position_spread_m

        state = SystemState(
            x=x_ref.x + np.random.uniform(-spread, spread),
            x_dot=0.0,
            alpha_x=np.random.uniform(-angle, angle),
            alpha_x_dot=0.0,
            y=x_ref.y + np.random.uniform(-spread, spread),
            y_dot=0.0,
            alpha_y=np.random.uniform(-angle, angle),
            alpha_y_dot=0.0,
        )

        cmd = TableCommand(x_ref.x, x_ref.y)

        return state, cmd


class RealTimeEngine(SimulationEngine):
    pass  # same execution, different runner internally if needed
