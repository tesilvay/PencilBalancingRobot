import numpy as np

from core.sim_types import (
    BenchmarkSummary,
)

def summarize(results):
    stability_rate = sum(r.stabilized for r in results) / len(results)

    settling_times = [r.settling_time for r in results if r.settling_time is not None]
    avg_settling = np.mean(settling_times) if settling_times else None

    max_acc = max(r.max_acc for r in results)
    avg_acc = np.mean([r.max_acc for r in results])

    return BenchmarkSummary(
        stability_rate=stability_rate,
        avg_settling_time=avg_settling,
        max_acc=max_acc,
        avg_acc=avg_acc,
    )

def print_summary(summary):
    print(f"Stability rate: {summary.stability_rate * 100:.1f}%")
    print(f"Avg settling: {summary.avg_settling_time:.2f}")
    print(f"Max acc: {summary.max_acc:.2f}")
    print(f"Avg acc: {summary.avg_acc:.2f}")
 