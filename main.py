# main.py
import argparse
from experiments.experiments import SingleExperiment, MonteCarloExperiment, BenchmarkExperiment, RealExperiment, WorkspaceSweepExperiment
from experiments.utils import print_summary
from simulation.engine import RealTimeEngine, SimulationEngine
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
    StopPolicy
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
                
                servo=True,
                servo_port=None,#"/dev/ttyUSB0", # None uses a mock controller
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
                realtimerender=True,
                total_time=5.0,  # 5s for single-run validation
                dt = 0.001,
                stability_tolerance=0.3,  # max |angle| rad: ~0.05 strict standing; ~0.3 at least upright (fell vs not)
                estimator_lpf_alpha=None,  # None = 0.95; 0.99 for lower phase lag (real-time)
                initial_angle_spread_deg=12,
                initial_position_spread_m=0.020,
            ),
        ),
        camera_params=CameraParams(xr=0.170, yr=0.176),
        default_variant=BenchmarkVariant(
            controller_type="lqr",
            estimator_type="kalman",
            noise_std=0.01,
            delay_steps=1,
        ),
    )


    if mode == "single":
        engine = SimulationEngine(stop_policy = StopPolicy.FIXED_TIME)
        experiment = SingleExperiment(engine)

    elif mode == "real":
        engine = RealTimeEngine(stop_policy = StopPolicy.INFINITE)
        experiment = RealExperiment(engine)

    elif mode == "montecarlo":
        engine = SimulationEngine(stop_policy = StopPolicy.EARLY_STOP)
        experiment = MonteCarloExperiment(engine, n_trials=200)

    elif mode == "benchmark":
        engine = SimulationEngine(stop_policy = StopPolicy.EARLY_STOP)
        experiment = BenchmarkExperiment(
            engine,
            variants=build_default_variants(),
            n_trials=200
        )

    elif mode == "sweep":
        engine = SimulationEngine(stop_policy = StopPolicy.EARLY_STOP)
        experiment = WorkspaceSweepExperiment(
            engine,
            min_diameter_mm=40,
            max_diameter_mm=80,
            n_sizes=20,
            n_trials=200,
        )

    elif mode == "graph":
        run_full_analysis()
        return

    else:
        raise ValueError(f"Unknown mode: {mode}")


    result = experiment.run(setup)
    
    # output

    if mode == "benchmark":
        save_benchmark_results(result)

    elif mode in ["single", "montecarlo", "real"]:
        print_summary(result)

    elif mode == "sweep":
        pass  # already plotted internally


def build_default_variants():
    controllers = ["lqr", "pole"]
    estimators = ["lpf", "kalman"]
    noises = [0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
    delays = [1]

    variants = []

    for c in controllers:
        for e in estimators:
            for n in noises:
                for d in delays:
                    variants.append(
                        BenchmarkVariant(
                            controller_type=c,
                            estimator_type=e,
                            noise_std=n,
                            delay_steps=d,
                        )
                    )

    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        default="single",
                        choices=["single",
                                 "real",
                                 "montecarlo",
                                 "benchmark",
                                 "graph",
                                 "sweep"])
    args = parser.parse_args()

    main(args.mode)