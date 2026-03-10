# benchmark.py
import numpy as np
import sys
import time
from simulation.simulation_runner import run_simulation
from visualization.visualizer3d import Visualizer3D
from core.sim_types import TrialMetrics, BenchmarkSummary, SystemState
from analysis.graphing import plot_state_history


class ProgressBar:
    def __init__(self, total, width=30, prefix=""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0

    def update(self, step):
        self.current = step
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "-" * (self.width - filled)

        percent = progress * 100

        sys.stdout.write(
            f"\r{self.prefix} |{bar}| {percent:6.2f}% ({self.current}/{self.total})"
        )
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def evaluate_stability(result, dt, tol=0.02, hold_time=0.2):
    alpha_x = result.state_history[:, 2]
    alpha_y = result.state_history[:, 6]

    inside = (np.abs(alpha_x) < tol) & (np.abs(alpha_y) < tol)
    hold_steps = int(hold_time / dt)

    for i in range(len(inside) - hold_steps):
        if np.all(inside[i:i + hold_steps]):
            return True, i * dt

    return False, None


def run_region_trials(
    params,
    plant,
    controller,
    vision=None,
    estimator=None,
    mech=None,
    actuator=None,
    dt=0.001,
    total_time=3.0,
    n_trials=100,
    show_progress=False,
    progress_prefix="",
    x_ref=None,
    realtime=False,
):
    results = []

    if show_progress:
        bar = ProgressBar(n_trials, prefix=progress_prefix)
    
    for trial in range(n_trials):

        if estimator is not None:
            estimator.reset()
        if vision is not None:
            vision.reset()

        initial_state = SystemState(
            x=x_ref.x,
            x_dot=0.0,
            alpha_x=np.random.uniform(-0.2, 0.2),
            alpha_x_dot=0.0,
            y=x_ref.y,
            y_dot=0.0,
            alpha_y=np.random.uniform(-0.2, 0.2),
            alpha_y_dot=0.0
        )


        sim_result = run_simulation(
            params=params,
            initial_state=initial_state,
            total_time=total_time,
            dt=dt,
            plant=plant,
            controller=controller,
            vision=vision,
            estimator=estimator,
            actuator=actuator,
            realtime=realtime,
        )
        
        if show_progress:
            bar.update(trial + 1)
        
        if n_trials == 1: # only render if we use the single mode
            #plot_state_history(sim_result.state_history, x_ref)
            viz = Visualizer3D(sim_result.state_history, dt=0.001, mech=mech, params=params)
            viz.render_video(video_speed=1, save_video=params.save_video)
            
            

        stabilized, settling_time = evaluate_stability(sim_result, dt)
        max_acc = np.max(np.abs(sim_result.acc_history))
        
        if max_acc > 200:
            print("Large acc trial initial state:", initial_state)

        results.append(
            TrialMetrics(
                stabilized=stabilized,
                settling_time=settling_time,
                max_acc=max_acc
            )
        )
    if show_progress:
        bar.finish()
    return results


def summarize_results(results: list[TrialMetrics]) -> BenchmarkSummary:

    stability_rate = sum(r.stabilized for r in results) / len(results)

    settling_times = [
        r.settling_time
        for r in results
        if r.settling_time is not None
    ]

    avg_settling = np.mean(settling_times) if settling_times else None

    max_acc = max(r.max_acc for r in results)
    avg_acc = np.mean([r.max_acc for r in results])

    return BenchmarkSummary(
        stability_rate=stability_rate,
        avg_settling_time=avg_settling,
        max_acc=max_acc,
        avg_acc=avg_acc
    )