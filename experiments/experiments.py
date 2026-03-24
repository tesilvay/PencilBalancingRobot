# experiments.py

import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from experiments.utils import summarize

from simulation.metrics import Metrics
from visualization.progress import ConsoleProgress
from core.sim_types import (
    BenchmarkResult,
    ExperimentSetup,
)

class Experiment(ABC):
    def __init__(self, engine, progress=ConsoleProgress()):
        self.engine = engine
        self.evaluator = Metrics()
        self.progress = progress

    @abstractmethod
    def run(self, setup: ExperimentSetup):
        pass

# =========================================================
# Concrete Experiments
# =========================================================

class SingleExperiment(Experiment):
    def run(self, setup):
        self.progress.start(1, "Single run")
        
        result = self.engine.run(setup)
        metrics = self.evaluator.evaluate(result)
        
        self.progress.update(1)
        self.progress.finish()
        
        self._animate_experiment(setup.params, result)
        
        return summarize([metrics])

    def _build_mech_history(self, state_history, mech):
    
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

    def _animate_experiment(self, params, sim_result):
        from core.system_builder import build_mechanism
        from visualization.visualizer3d import Visualizer3D
        mech = build_mechanism(params)

        mech_history = self._build_mech_history(
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

class RealExperiment(Experiment):
    def run(self, setup):
      
        result = self.engine.run(setup)
        metrics = self.evaluator.evaluate(result)
        
        return summarize([metrics])

class MonteCarloExperiment(Experiment):
    def __init__(self, engine, n_trials=200):
        super().__init__(engine)
        self.n_trials = n_trials

    def run(self, setup):
        results = []
        
        self.progress.start(self.n_trials, "Monte Carlo")
        
        for i in range(self.n_trials):
            m = self.evaluator.evaluate(self.engine.run(setup))
            results.append(m)

            self.progress.update(i + 1)

        self.progress.finish()
            
        return summarize(results)

class BenchmarkExperiment(Experiment):
    def __init__(self, engine, variants, n_trials=200):
        super().__init__(engine)
        self.variants = variants
        self.n_trials = n_trials

    def run(self, setup):
        all_results = []

        for idx, variant in enumerate(self.variants):

            label = f"Variant {idx+1}/{len(self.variants)}"

            self.progress.start(self.n_trials, label)

            variant_setup = ExperimentSetup(
                params=setup.params,
                camera_params=setup.camera_params,
                default_variant=variant,
            )

            results = []
            for i in range(self.n_trials):
                m = self.evaluator.evaluate(
                    self.engine.run(variant_setup)
                )
                results.append(m)


                self.progress.update(i + 1)

            self.progress.finish()

            summary = summarize(results)

            all_results.append(
                BenchmarkResult(
                    params=setup.params,
                    variant=variant,
                    summary=summary,
                )
            )

        return all_results

class WorkspaceSweepExperiment(Experiment):
    def __init__(
        self,
        engine,
        min_diameter_mm: float,
        max_diameter_mm: float,
        n_sizes: int = 5,
        n_trials: int = 200,
    ):
        super().__init__(engine)

        self.min_d = min_diameter_mm
        self.max_d = max_diameter_mm
        self.n_sizes = n_sizes
        self.n_trials = n_trials

    def setup_with_workspace_radius(self, setup, r_m):
        new_setup = setup
        new_setup.params.workspace.safe_radius = r_m
        return new_setup

    def run(self, setup: ExperimentSetup):

        diameters_mm = np.linspace(self.min_d, self.max_d, self.n_sizes)
        radii_mm = diameters_mm / 2

        stability_rates = []
        avg_accs = []

        for idx, r_mm in enumerate(radii_mm):

            r_m = r_mm / 1000.0

            # --- SAFE: create new setup ---
            variant_setup = self.setup_with_workspace_radius(setup, r_m)

            if self.progress:
                label = f"Radius {r_mm:.1f} mm ({idx+1}/{len(radii_mm)})"
                self.progress.start(self.n_trials, label)

            results = []

            for i in range(self.n_trials):
                m = self.evaluator.evaluate(
                    self.engine.run(variant_setup)
                )
                results.append(m)

                if self.progress:
                    self.progress.update(i + 1)

            if self.progress:
                self.progress.finish()

            summary = summarize(results)

            stability_rates.append(summary.stability_rate * 100)
            avg_accs.append(summary.avg_acc)

            print(
                f"Radius {r_mm:.1f} mm -> "
                f"Stability {summary.stability_rate*100:.1f}% | "
                f"Avg Acc {summary.avg_acc:.2f}"
            )

        data = np.column_stack((radii_mm, stability_rates, avg_accs))

        self._plot(data, setup)

        return data

    def _plot(self, data, setup):

        radii = data[:, 0]
        stability = data[:, 1]
        avg_acc = data[:, 2]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Workspace Radius (mm)")
        ax1.set_ylabel("Stability Rate (%)")
        ax1.plot(radii, stability, marker="o", linewidth=2)
        ax1.set_ylim(0, 110)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Average Acceleration (m/s²)")
        ax2.plot(radii, avg_acc, marker="s", linewidth=2)

        plt.title("Workspace Radius vs Control Performance")
        plt.grid(True)

        variant = setup.default_variant

        variant_text = (
            f"Controller: {variant.controller_type}\n"
            f"Estimator: {variant.estimator_type}\n"
            f"Noise σ: {variant.noise_std}\n"
            f"Delay: {variant.delay_steps} steps\n"
            f"Trials per radius: {self.n_trials}"
        )

        ax1.text(
            0.02,
            0.98,
            variant_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()
