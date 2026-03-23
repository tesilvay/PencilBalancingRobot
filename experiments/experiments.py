# experiment.py
import numpy as np
from experiments.monte_carlo import run_region_trials, summarize_results
from core.sim_types import BenchmarkResult, BenchmarkVariant, ExperimentSetup
from analysis.workspace_plotter import plot_workspace_results



def format_summary(summary, variant, params):
    
    # Workspace size
    diameter_cm = params.workspace.safe_radius * 2 * 100
    
    
    stability_pct = summary.stability_rate * 100

    settling = (
        f"{summary.avg_settling_time:.3f} s"
        if summary.avg_settling_time is not None
        else "Did not settle"
    )

    return (
        f"  Controller         : {variant.controller_type}\n"
        f"  Estimator          : {variant.estimator_type}\n"
        f"  Noise              : {variant.noise_std}\n"
        f"  Delay              : {variant.delay_steps}\n"
        f"  Workspace diameter : {diameter_cm:.1f} cm\n"
        f"\n"
        f"  Stability rate     : {stability_pct:.1f}%\n"
        f"  Avg settling time  : {settling}\n"
        f"  Max acceleration   : {summary.max_acc:.2f} m/s²\n"
        f"  Avg acceleration   : {summary.avg_acc:.2f} m/s²"
    )


def run_single(setup: ExperimentSetup):
    
    results = run_region_trials(
        setup=setup,
        n_trials=1,
    )

    return summarize_results(results)


def run_benchmark_single(setup: ExperimentSetup):
    
    results = run_region_trials(
        setup=setup,
        n_trials=200,
        show_progress=True,
        progress_prefix="Trial",
    )

    return summarize_results(results)


def run_benchmark_all(setup: ExperimentSetup):

    controllers = ["lqr", "pole"]
    estimators = ["lpf", "kalman"]
    noises = [0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
    delays = [1]

    all_results = []
    total_configs = len(controllers) * len(estimators) * len(noises) * len(delays)
    config_index = 0

    for c in controllers:
        for e in estimators:
            for n in noises:
                for d in delays:
                    
                    config_index += 1
                    print(f"\nEpoch {config_index}/{total_configs}")
                    print(f"Controller={c}, Estimator={e}, Noise={n}, Delay={d}")

                    variant = BenchmarkVariant(
                        controller_type=c,
                        estimator_type=e,
                        noise_std=n,
                        delay_steps=d
                    )

                    # Temporary setup with this variant for run_benchmark_single
                    variant_setup = ExperimentSetup(
                        params=setup.params,
                        camera_params=setup.camera_params,
                        default_variant=variant,
                    )
                    metrics = run_benchmark_single(variant_setup)

                    all_results.append(
                        BenchmarkResult(
                            params=setup.params,
                            variant=variant,
                            summary=metrics
                        )
                    )

    return all_results


def sweep_workspace(
    setup: ExperimentSetup,
    workspace_min_diameter_mm: float,
    workspace_max_diameter_mm: float,
    n_sizes: int = 5,
):
    """
    Sweep workspace sizes and benchmark controller performance.
    """

    diameters_mm = np.linspace(
        workspace_min_diameter_mm,
        workspace_max_diameter_mm,
        n_sizes,
    )

    radii_mm = diameters_mm / 2

    stability_rates = []
    avg_accs = []

    for r_mm in radii_mm:

        r_m = r_mm / 1000.0

        # Update workspace limits (plant uses safe_radius for circular workspace)
        setup.params.workspace.safe_radius = r_m

        summary = run_benchmark_single(setup)

        stability_rates.append(summary.stability_rate * 100)
        avg_accs.append(summary.avg_acc)

        print(f"Radius {r_mm:.1f} mm -> "
              f"Stability {summary.stability_rate*100:.1f}% | "
              f"Avg Acc {summary.avg_acc:.2f}")

    data = np.column_stack((radii_mm, stability_rates, avg_accs))

    plot_workspace_results(data, setup.default_variant)

    return data
