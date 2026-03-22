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
                tau=0.04,
                zeta=0.7,
                num_states=8,
                max_acc=9.81 * 3,
            ),
            workspace=WorkspaceParams(
                x_ref=0.0,
                y_ref=0.0,
                safe_radius=0.108,  # min is 0.031 for 100% stability
            ),
            mechanism=MechanismParams( # mechanism in mm
                O=(128.77, 178.13),
                B=(101.77, 210.13),

                la=175,
                lb=175,
            ),
            hardware=HardwareParams(
                
                servo=False,
                servo_port="/dev/ttyUSB0", # None uses a mock controller
                servo_frequency=250,
                
                vision_mode = "sim_analytic", # "real_dvs" | "sim_dvs" | "sim_analytic"
                
                dvs_use_regression=True,
                
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
                realtimerender=False,
                total_time=5.0,  # 5s for single-run validation
                stability_tolerance=0.3,  # max |angle| rad: ~0.05 strict standing; ~0.3 at least upright (fell vs not)
                estimator_lpf_alpha=None,  # None = 0.95; 0.99 for lower phase lag (real-time)
                initial_angle_spread_deg=10,
                initial_position_spread_m=0.050,
            ),
        ),
        camera_params=CameraParams(xr=0.170, yr=0.176),
        default_variant=BenchmarkVariant(
            controller_type="lqr",
            estimator_type="kalman",
            noise_std=0.001,
            delay_steps=1,
        ),
    )

    if mode == "single":
        summary = run_single(setup)
        print("\n=== Single Trial ===")
        print(format_summary(summary, setup.default_variant, setup.params))

    elif mode == "benchmark":
        # Shorter total_time for faster benchmark (200 trials)
        setup.params.run.total_time = 2.0
        summary = run_benchmark_single(setup)
        print("\n=== Monte Carlo Benchmark ===")
        print(format_summary(summary, setup.default_variant, setup.params))

    elif mode == "sweep":
        sweep_workspace(
            setup,
            workspace_min_diameter_mm=40,
            workspace_max_diameter_mm=80,
            n_sizes=20,
        )

    elif mode == "benchmark_all":
        setup.params.run.total_time = 2.0
        results = run_benchmark_all(setup)
        save_benchmark_results(results)

    elif mode == "graph":
        run_full_analysis()

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        default="single",
                        choices=["single",
                                 "benchmark",
                                 "benchmark_all",
                                 "graph",
                                 "sweep"])
    args = parser.parse_args()

    main(args.mode)