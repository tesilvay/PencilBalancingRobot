# experiment.py
from dataclasses import dataclass
import numpy as np
from controller import NullController, PolePlacementController, LQRController, build_lqr_weights
from estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from vision import VisionSystem
from model import BuildLinearModel
from benchmark import run_region_trials, summarize_results
from sim_types import PhysicalParams, CameraParams, BenchmarkSummary, BenchmarkResult, ExperimentConfig

def format_summary(summary):
    stability_pct = summary.stability_rate * 100

    settling = (
        f"{summary.avg_settling_time:.3f} s"
        if summary.avg_settling_time is not None
        else "Did not settle"
    )

    return (
        f"  Stability rate     : {stability_pct:.1f}%\n"
        f"  Avg settling time  : {settling}\n"
        f"  Max acceleration   : {summary.max_acc:.2f} m/s²\n"
        f"  Avg acceleration   : {summary.avg_acc:.2f} m/s²"
    )


def build_system(config, params, camera_params):

    A, B = BuildLinearModel(params)

    # Controller
    if config.controller_type == "pole":
        poles = [-8, -10, -12, -14] * 2
        controller = PolePlacementController(A, B, poles)

    elif config.controller_type == "lqr":
        Q, R = build_lqr_weights(
            x_max=0.05,
            xdot_max=0.5,
            alpha_max=0.2,
            alphadot_max=2.0,
            u_max=0.05,
            angle_importance=config.angle_importance,
            effort_scale=config.effort_scale
        )
        controller = LQRController(A, B, Q, R)

    else:
        controller = NullController()

    # Estimator
    estimator = None
    vision = None

    if config.estimator_type is not None:

        vision = VisionSystem(
            camera_params,
            noise_std=config.noise_std,
            delay_steps=config.delay_steps
        )

        if config.estimator_type == "fd":
            estimator = FiniteDifferenceEstimator()

        elif config.estimator_type == "lpf":
            estimator = LowPassFiniteDifferenceEstimator()

        elif config.estimator_type == "kalman":
            Qk = np.eye(8) * 1e-6
            Rk = np.eye(4) * config.noise_std**2
            estimator = KalmanEstimator(A, dt=0.001, Q=Qk, R=Rk)

    return controller, vision, estimator


def run_single(config, params, camera_params):

    controller, vision, estimator = build_system(config, params, camera_params)

    results = run_region_trials(
        params=params,
        controller=controller,
        vision=vision,
        estimator=estimator,
        n_trials=1
    )

    return summarize_results(results)


def run_benchmark_single(config, params, camera_params):

    controller, vision, estimator = build_system(config, params, camera_params)

    results = run_region_trials(
        params=params,
        controller=controller,
        vision=vision,
        estimator=estimator,
        n_trials=200,
        show_progress=True,
        progress_prefix="Trial"
    )

    return summarize_results(results)


def run_benchmark_all(params, camera_params):

    controllers = ["pole", "lqr"]
    estimators = [None, "fd", "lpf", "kalman"]
    noises = [0, 1e-4, 5e-4]
    delays = [0, 2]

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

                    metrics = run_benchmark_single(config, params, camera_params)

                    all_results.append(
                        BenchmarkResult(
                            config=config,
                            summary=metrics
                        )
                    )

    return all_results