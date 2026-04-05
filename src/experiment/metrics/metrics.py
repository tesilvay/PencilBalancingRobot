from dataclasses import dataclass

import numpy as np

from src.experiment.logger import SimulationResult


@dataclass
class TrialMetrics:
    stabilized: bool
    settling_time: float | None
    max_acc: float
    avg_innovation: np.ndarray | None = None


@dataclass
class BenchmarkSummary:
    stability_rate: float
    avg_settling_time: float | None
    max_acc: float
    avg_acc: float
    avg_innovation: np.ndarray | None = None


class Metrics:
    def evaluate(self, result: SimulationResult) -> TrialMetrics:
        innov = result.innovation_history
        if innov is None or innov.size == 0:
            avg_innov = None
        else:
            avg_innov = np.mean(np.abs(innov), axis=0)

        term = result.terminal
        return TrialMetrics(
            stabilized=term.stabilized if term else False,
            settling_time=term.settling_time if term else None,
            max_acc=float(np.max(np.abs(result.acc_history)))
            if result.acc_history.size
            else 0.0,
            avg_innovation=avg_innov,
        )

    def summarize(self, trial_metrics: list[TrialMetrics]) -> BenchmarkSummary:
        stability_rate = sum(r.stabilized for r in trial_metrics) / len(trial_metrics)

        settling_times = [
            r.settling_time for r in trial_metrics if r.settling_time is not None
        ]
        avg_settling = np.mean(settling_times) if settling_times else None

        max_acc = max(r.max_acc for r in trial_metrics)
        avg_acc = np.mean([r.max_acc for r in trial_metrics])

        with_innov = [r for r in trial_metrics if r.avg_innovation is not None]
        if with_innov:
            avg_innovation = np.mean([r.avg_innovation for r in with_innov], axis=0)
        else:
            avg_innovation = None

        return BenchmarkSummary(
            stability_rate=stability_rate,
            avg_settling_time=avg_settling,
            max_acc=max_acc,
            avg_acc=avg_acc,
            avg_innovation=avg_innovation,
        )

    def print_summary(self, summary: BenchmarkSummary) -> None:
        avg_settling_time = summary.avg_settling_time or -1
        print(
            f"\n\nStability rate: {summary.stability_rate * 100:.1f}%\n"
            f"Avg settling: {avg_settling_time:.2f}\n"
            f"Max acc: {summary.max_acc:.2f}\n"
            f"Avg acc: {summary.avg_acc:.2f}\n"
        )
        if summary.avg_innovation is None:
            print("Innovation: (no history)\n")
            return

        m = summary.avg_innovation
        # Channels match z = [px, ax, py, ay] (positions m, angles rad).
        print(
            "mean |innovation| (meas. channels):\n"
            f"{'channel':<10} {'px (m)':>12} {'ax (deg)':>12} {'py (m)':>12} {'ay (deg)':>12}\n"
            f"{'-'*58}\n"
            f"{'avg |·|':<10} {m[0]:>12.4e} {np.rad2deg(m[1]):>12.4f} "
            f"{m[2]:>12.4e} {np.rad2deg(m[3]):>12.4f}\n"
        )
