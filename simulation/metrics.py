from core.sim_types import SimulationResult, TrialMetrics, SystemState
import numpy as np

class Metrics:
    def _turn_to_system_state(self, vector):
        return SystemState(
            x=vector[0],
            x_dot=vector[1],
            alpha_x=vector[2],
            alpha_x_dot=vector[3],
            y=vector[4],
            y_dot=vector[5],
            alpha_y=vector[6],
            alpha_y_dot=vector[7],
        )
    def evaluate(self, result: SimulationResult) -> TrialMetrics:
        return TrialMetrics(
            stabilized=result.terminal.stabilized,
            settling_time=result.terminal.settling_time,
            max_acc=np.max(np.abs(result.acc_history)),
            avg_state_est_err=np.mean(np.abs(result.state_est_err_history), axis=0),
        )

