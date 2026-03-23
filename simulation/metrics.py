from core.sim_types import SimulationResult, TrialMetrics
import numpy as np

class Metrics:
    def evaluate(self, result: SimulationResult) -> TrialMetrics:
        return TrialMetrics(
            stabilized=result.terminal.stabilized,
            settling_time=result.terminal.settling_time,
            max_acc=np.max(np.abs(result.acc_history))
        )

