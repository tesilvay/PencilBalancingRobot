# main.py
import argparse
from experiments.experiments import run_single, run_benchmark_single, run_benchmark_all, format_summary, sweep_workspace
from core.sim_types import (
    PhysicalParams,
    PlantParams,
    WorkspaceParams,
    MechanismParams,
    HardwareParams,
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
                x_ref=-0.00993,
                y_ref=0.01553,
                safe_radius=0.040,  # min is 0.031 for 100% stability
            ),
            mechanism=MechanismParams(
                O=(83, 57),
                B=(61, 88),
                la=77,
                lb=69.6,
            ),
            hardware=HardwareParams(
                servo=True,
                servo_port=None,
                dvs_cam=False,
                dvs_cam_x_port=None,
                dvs_cam_y_port=None,
            ),
            run=RunParams(
                save_video=False,
                realtimerender=True,
                total_time=1.0,
                stability_tolerance=0.05,
            ),
        ),
        camera_params=CameraParams(xr=0.3, yr=0.3),
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