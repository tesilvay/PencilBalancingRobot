# main.py
import argparse
from experiments.experiments import run_single, run_benchmark_single, run_benchmark_all, format_summary, sweep_workspace
from core.sim_types import (
    PhysicalParams,
    PlantParams,
    WorkspaceParams,
    MechanismParams,
    HardwareParams,
    HoughTrackerParams,
    RunParams,
    CameraParams,
    BenchmarkVariant,
    ExperimentSetup,
)
from analysis.graphing import run_full_analysis
from utils.io_utils import save_benchmark_results


def main(mode):

    setup = ExperimentSetup(
        params=PhysicalParams(
            plant=PlantParams(
                g=9.81,
                com_length=0.1,
                tau=0.02,
                zeta=0.7,
                num_states=8,
                max_acc=9.81 * 9,
            ),
            workspace=WorkspaceParams(
                x_ref=-0.00466,
                y_ref=0.00955,
                safe_radius=0.020,  # min is 0.031 for 100% stability
            ),
            mechanism=MechanismParams(
                O=(85.91, 57.86),
                B=(60.10, 87.07),
                la=76.84,
                lb=66.83,
            ),
            hardware=HardwareParams(
                
                servo=True,
                servo_port="/dev/ttyUSB0", # None uses a mock controller
                servo_frequency=250,
                
                dvs_cam=True,
                
                dvs_cam_x_port=None,  # or serials for real DVS; None uses discovery
                dvs_cam_y_port=None,
                dvs_algo="hough",  # "hough" | "sam"
                dvs_sam_noise_filter_duration_ms=5,  # None = no filter; 5–10 for low-latency
                dvs_hough=HoughTrackerParams(
                    mixing_factor=0.02,  # Hough only: higher tracks faster but gets noisier; 0.01-0.05 is a good starting range.
                    inlier_stddev_px=4.0,  # Hough only: Gaussian inlier width in pixels; 3-6 px is typical, larger admits more background motion.
                    min_determinant=1e-6,  # Hough only: reject unstable solves when the quadratic is near-singular; usually leave at 1e-6.
                ),
            ),
            run=RunParams(
                save_video=False,
                realtimerender=True,
                total_time=5.0,  # 5s for single-run validation
                stability_tolerance=0.05,  # 5% stability should be possible
                estimator_lpf_alpha=None,  # None = 0.95; 0.99 for lower phase lag (real-time)
            ),
        ),
        camera_params=CameraParams(xr=0.170, yr=0.176),
        default_variant=BenchmarkVariant(
            controller_type="lqr",
            estimator_type="lpf",
            noise_std=0.0001,
            delay_steps=1,
        ),
    )

    if mode == "single":
        summary = run_single(setup)
        print("\n=== Single Trial ===")
        print(format_summary(summary, setup.default_variant, setup.params))

    elif mode == "benchmark_single":
        # Shorter total_time for faster benchmark (200 trials)
        setup.params.run.total_time = 2.0
        summary = run_benchmark_single(setup)
        print("\n=== Monte Carlo Benchmark ===")
        print(format_summary(summary, setup.default_variant, setup.params))

    elif mode == "sweep_workspace":
        sweep_workspace(
            setup,
            workspace_min_diameter_mm=40,
            workspace_max_diameter_mm=80,
            n_sizes=20,
        )

    elif mode == "benchmark_all_configs":
        results = run_benchmark_all(setup)
        save_benchmark_results(results)

    elif mode == "graph_results":
        run_full_analysis()

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        default="single",
                        choices=["single",
                                 "benchmark_single",
                                 "benchmark_all_configs",
                                 "graph_results",
                                 "sweep_workspace"])
    args = parser.parse_args()

    main(args.mode)