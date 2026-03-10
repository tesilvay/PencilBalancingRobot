# experiment.py
import numpy as np
from experiments.monte_carlo import run_region_trials, summarize_results
from core.sim_types import BenchmarkResult, ExperimentConfig
from analysis.workspace_plotter import plot_workspace_results
from system_builder import build_system


def format_summary(summary, config, params):
    
    # Workspace size
    diameter_cm = params.safe_radius * 2 * 100
    
    
    stability_pct = summary.stability_rate * 100

    settling = (
        f"{summary.avg_settling_time:.3f} s"
        if summary.avg_settling_time is not None
        else "Did not settle"
    )

    return (
        f"  Controller         : {config.controller_type}\n"
        f"  Estimator          : {config.estimator_type}\n"
        f"  Noise              : {config.noise_std}\n"
        f"  Delay              : {config.delay_steps}\n"
        f"  Workspace diameter : {diameter_cm:.1f} cm\n"
        f"\n"
        f"  Stability rate     : {stability_pct:.1f}%\n"
        f"  Avg settling time  : {settling}\n"
        f"  Max acceleration   : {summary.max_acc:.2f} m/s²\n"
        f"  Avg acceleration   : {summary.avg_acc:.2f} m/s²"
    )


def run_single(config, params, camera_params, x_ref=None):

    plant,controller, vision, estimator, mech, actuator, visualizer = build_system(config, params, camera_params, x_ref=x_ref)

    results = run_region_trials(
        params=params,
        plant=plant,
        controller=controller,
        vision=vision,
        estimator=estimator,
        mech=mech,
        actuator=actuator,
        visualizer=visualizer,
        n_trials=1,
        x_ref=x_ref,
        realtime=True,
    )

    return summarize_results(results)


def run_benchmark_single(config, params, camera_params, x_ref=None):
    
    # prevents dvs from starting in the benchmark
    params.realtimerender = False

    plant, controller, vision, estimator, mech, _, _ = build_system(config, params, camera_params, x_ref=x_ref)

    results = run_region_trials(
        params=params,
        plant=plant,
        controller=controller,
        vision=vision,
        estimator=estimator,
        mech=mech,
        n_trials=200,
        show_progress=True,
        progress_prefix="Trial",
        x_ref=x_ref,
        realtime=False,
    )

    return summarize_results(results)


def run_benchmark_all(params, camera_params, x_ref=None):

    controllers = ["pole"]
    estimators = ["lpf"]
    noises = [0, 1e-3, 5e-3, 1e-2, 5e-2]
    delays = [0, 2, 10, 15]

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

                    config = ExperimentConfig(
                        controller_type=c,
                        estimator_type=e,
                        noise_std=n,
                        delay_steps=d
                    )

                    metrics = run_benchmark_single(config, params, camera_params, x_ref=x_ref)

                    all_results.append(
                        BenchmarkResult(
                            params=params,
                            config=config,
                            summary=metrics
                        )
                    )

    return all_results


def sweep_workspace(
    config,
    params,
    camera_params,
    workspace_min_diameter_mm,
    workspace_max_diameter_mm,
    n_sizes=5
):
    """
    Sweep workspace sizes and benchmark controller performance.
    """

    diameters_mm = np.linspace(workspace_min_diameter_mm,
                               workspace_max_diameter_mm,
                               n_sizes)

    radii_mm = diameters_mm / 2

    stability_rates = []
    avg_accs = []

    for r_mm in radii_mm:

        r_m = r_mm / 1000.0

        # Update workspace limits
        params.x_min = -r_m
        params.x_max = r_m
        params.y_min = -r_m
        params.y_max = r_m

        summary = run_benchmark_single(config, params, camera_params)

        stability_rates.append(summary.stability_rate * 100)
        avg_accs.append(summary.avg_acc)

        print(f"Radius {r_mm:.1f} mm -> "
              f"Stability {summary.stability_rate*100:.1f}% | "
              f"Avg Acc {summary.avg_acc:.2f}")

    data = np.column_stack((radii_mm, stability_rates, avg_accs))

    plot_workspace_results(data, config)

    return data
