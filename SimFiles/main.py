# main.py
import argparse
from experiment import run_single, run_benchmark_single, run_benchmark_all, format_summary
from sim_types import PhysicalParams, CameraParams, ExperimentConfig


def main(mode):

    params = PhysicalParams(
        g=9.81,
        com_length=0.1,
        tau=0.04,
        zeta=0.7,
        num_states=8
    )

    camera_params = CameraParams(xr=0.3, yr=0.3)

    default_config = ExperimentConfig(
        controller_type="pole",
        estimator_type=None,
        noise_std=0.0,
        delay_steps=0
    )

    if mode == "single":
        summary = run_single(default_config, params, camera_params)

        print("\n=== Single Trial ===")
        print(format_summary(summary))


    elif mode == "benchmark_single":
        summary = run_benchmark_single(default_config, params, camera_params)

        print("\n=== Monte Carlo Benchmark ===")
        print(format_summary(summary))

    elif mode == "benchmark_all_configs":

        results = run_benchmark_all(params, camera_params)

        print("\n=== Full Benchmark Sweep ===\n")

        for i, result in enumerate(results, 1):

            config = result.config
            summary = result.summary

            print(f"--- Configuration {i} ---")
            print(f"  Controller        : {config.controller_type}")
            print(f"  Estimator         : {config.estimator_type}")
            print(f"  Noise std         : {config.noise_std}")
            print(f"  Delay steps       : {config.delay_steps}")
            print(format_summary(summary))
            print()

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        default="single",
                        choices=["single",
                                 "benchmark_single",
                                 "benchmark_all_configs"])
    args = parser.parse_args()

    main(args.mode)