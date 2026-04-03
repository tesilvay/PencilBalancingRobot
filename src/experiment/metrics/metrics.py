from dataclasses import dataclass

import numpy as np

from src.experiment.logger import SimulationResult
from src.shared import SystemState


@dataclass
class TrialMetrics:
    stabilized: bool
    settling_time: float | None
    max_acc: float
    avg_state_est_err: np.ndarray | None = None


@dataclass
class BenchmarkSummary:
    stability_rate: float
    avg_settling_time: float | None
    max_acc: float
    avg_acc: float
    avg_state_est_err: np.ndarray | None = None




class Metrics:
    def _turn_to_system_state(self, vector):
        return SystemState(
            px=vector[0],
            vx=vector[1],
            ax=vector[2],
            wx=vector[3],
            py=vector[4],
            vy=vector[5],
            ay=vector[6],
            wy=vector[7],
        )
    def evaluate(self, result: SimulationResult) -> TrialMetrics:
        return TrialMetrics(
            stabilized=result.terminal.stabilized,
            settling_time=result.terminal.settling_time,
            max_acc=np.max(np.abs(result.acc_history)),
            avg_state_est_err=np.mean(np.abs(result.state_est_err_history), axis=0),
        )

    def summarize(results):
        stability_rate = sum(r.stabilized for r in results) / len(results)

        settling_times = [r.settling_time for r in results if r.settling_time is not None]
        avg_settling = np.mean(settling_times) if settling_times else None

        max_acc = max(r.max_acc for r in results)
        avg_acc = np.mean([r.max_acc for r in results])
        
        avg_state_est_err = np.mean([r.avg_state_est_err for r in results], axis=0)

        return BenchmarkSummary(
            stability_rate=stability_rate,
            avg_settling_time=avg_settling,
            max_acc=max_acc,
            avg_acc=avg_acc,
            avg_state_est_err=avg_state_est_err,
        )

    def turn_to_system_state(vector):
            return SystemState(
                px=vector[0],
                vx=vector[1],
                ax=vector[2],
                wx=vector[3],
                py=vector[4],
                vy=vector[5],
                ay=vector[6],
                wy=vector[7],
            )
            
    def print_summary(summary):
        avg_settling_time = summary.avg_settling_time or -1
        norm_err = summary.avg_state_est_err
        print(
            f"\n\nStability rate: {summary.stability_rate * 100:.1f}%\n"
            f"Avg settling: {avg_settling_time:.2f}\n"
            f"Max acc: {summary.max_acc:.2f}\n"
            f"Avg acc: {summary.avg_acc:.2f}\n\n"
            f"est err (% of scale):\n"
            f"{'var':<10} {'x':>8} {'y':>8}\n"
            f"{'-'*26}\n"
            f"{'pos':<10} {norm_err[0]:>+7.1f}% {norm_err[4]:>+7.1f}%\n"
            f"{'vel':<10} {norm_err[1]:>+7.1f}% {norm_err[5]:>+7.1f}%\n"
            f"{'angle':<10} {norm_err[2]:>+7.1f}% {norm_err[6]:>+7.1f}%\n"
            f"{'ang vel':<10} {norm_err[3]:>+7.1f}% {norm_err[7]:>+7.1f}%\n"
        )
 