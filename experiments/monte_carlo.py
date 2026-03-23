# benchmark.py
import numpy as np
import sys
import time
from visualization.visualizer3d import Visualizer3D
from core.sim_types import TrialMetrics, BenchmarkSummary, SystemState, make_reference_state, TableCommand
from analysis.graphing import plot_state_history
from system_builder import system_factory, runner_factory


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


def evaluate_stability(result, dt, tol=0.05, hold_time=1.0):
    alpha_x = result.state_history[:, 2]
    alpha_y = result.state_history[:, 6]

    inside = (np.abs(alpha_x) < tol) & (np.abs(alpha_y) < tol)
    hold_steps = int(hold_time / dt)

    for i in range(len(inside) - hold_steps):
        if np.all(inside[i:i + hold_steps]):
            return True, i * dt

    return False, None

def build_mech_history(state_history, mech):
 
    if mech is None:
        return None

    N = len(state_history)
    mech_history = np.full((N, 3, 2), np.nan, dtype=np.float64)

    for i in range(N):
        state = state_history[i]

        try:
            # state = [x, y, ...]
            x, y = state[0], state[4]

            _, _, A_g, C_g, P_g = mech.solve(
                np.array([x, y]) * 1000.0
            )

            mech_history[i, 0, :] = A_g
            mech_history[i, 1, :] = C_g
            mech_history[i, 2, :] = P_g

        except ValueError:
            # leave as NaN
            continue

    return mech_history

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

def sample_initial_state(params, x_ref):
    angle_rad = np.deg2rad(params.run.initial_angle_spread_deg)
    pos_spread = params.run.initial_position_spread_m

    return SystemState(
        x=x_ref.x + np.random.uniform(-pos_spread, pos_spread),
        x_dot=0.0,
        alpha_x=np.random.uniform(-angle_rad, angle_rad),
        alpha_x_dot=0.0,
        y=x_ref.y + np.random.uniform(-pos_spread, pos_spread),
        y_dot=0.0,
        alpha_y=np.random.uniform(-angle_rad, angle_rad),
        alpha_y_dot=0.0
    )

def run_region_trials(
    setup,
    n_trials=100,
    show_progress=False,
    progress_prefix="",
):
    
    variant = setup.default_variant
    params = setup.params
    camera_params = setup.camera_params
    
    results = []
    x_ref = make_reference_state(params.workspace)

    if show_progress:
        bar = ProgressBar(n_trials, prefix=progress_prefix)

    for trial in range(n_trials):

        # ---- 1. sample initial condition ----
        initial_state = sample_initial_state(params, x_ref)

        # ---- 2. build system (fresh each trial) ----
        system = system_factory(variant, params, camera_params)

        # ---- 3. build runner + logger ----
        runner, logger = runner_factory(params, system, n_trials)

        # ---- 4. initialize + run ----
        initial_command = TableCommand(x_ref.x, x_ref.y)
        runner.initialize(initial_state, initial_command)

        sim_result = runner.run()  # returns SimulationResult

        # ---- 5. progress ----
        if show_progress:
            bar.update(trial + 1)

        # ---- 6. optional rendering (single trial only) ----
        if n_trials == 1 and not params.run.realtimerender:
            
            from system_builder import build_mechanism
            mech = build_mechanism(params)

            mech_history = build_mech_history(
                sim_result.state_history,
                mech
            )

            viz = Visualizer3D(
                sim_result.state_history,
                dt=0.001,
                L=params.plant.com_length * 2,
                mech=mech,
                mech_history=mech_history,
                params=params,
                cmd_history=sim_result.cmd_history,
            )

            viz.render_video(
                video_speed=1,
                save_video=params.run.save_video
            )

        # ---- 7. metrics ----
        stabilized = runner.stop_condition.is_stabilized()
        _, settling_time = evaluate_stability(
            sim_result, dt=0.001, tol=params.run.stability_tolerance
        )

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