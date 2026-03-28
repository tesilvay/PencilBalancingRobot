# main.py
import argparse
import json

from experiments.experiments import (
    SingleExperiment,
    MonteCarloExperiment,
    BenchmarkExperiment,
    RealExperiment,
    WorkspaceSweepExperiment,
)
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
    StopPolicy,
)
from analysis.graphing import run_full_analysis
from utils.io_utils import save_benchmark_results
from UI.config_overrides import OverrideError, apply_overrides


# ----- Preset x experiment validation (from new_main) -----
VALID = {
    "sim": {"single", "montecarlo", "benchmark", "sweep"},
    "vision_real": {"single"},
    "actuation_real": {"single"},
    "real": {"single"},
}

DEFAULT_TRIALS = {
    "single": 1,
    "montecarlo": 200,
    "benchmark": 100,
    "sweep": 100,
}

# When --actuator is omitted, match new_main preset intent
PRESET_DEFAULT_ACTUATOR = {
    "sim": "sim",
    "vision_real": "sim",
    "actuation_real": "servo",
    "real": "servo",
}


def validate_preset_experiment(args: argparse.Namespace) -> None:
    if args.experiment not in VALID[args.preset]:
        raise ValueError(
            f"Experiment '{args.experiment}' not allowed in preset '{args.preset}'"
        )


def normalize_estimator(est: str | None) -> str | None:
    """CLI uses 'fde' (new_main); runtime uses 'fd' in system_builder."""
    if est == "fde":
        return "fd"
    return est


def resolve_effective_actuator(args: argparse.Namespace) -> str:
    if args.actuator is not None:
        return args.actuator
    return PRESET_DEFAULT_ACTUATOR[args.preset]


def apply_actuator_to_hardware(hw: HardwareParams, actuator: str) -> None:
    # sim: no hardware servo path; servo: real port; mock: ServoSystem with port=None
    if actuator == "sim":
        hw.servo = False
        hw.servo_port = None
    elif actuator == "servo":
        hw.servo = True
        hw.servo_port = "/dev/ttyUSB0"
    elif actuator == "mock":
        hw.servo = True
        hw.servo_port = None
    else:
        raise ValueError(f"Unknown actuator: {actuator}")


def _default_hardware() -> HardwareParams:
    return HardwareParams(
        servo=False,
        servo_port=None,
        vision_mode="sim_analytic",
        dvs_use_regression=False,
        dvs_algo="hough",
        sam_filter_ms=5,
        dvs_hough=HoughTrackerParams(
            mixing_factor=0.02,
            inlier_stddev_px=4.0,
            min_determinant=1e-6,
        ),
    )


def _default_run_offline() -> RunParams:
    return RunParams(
        save_video=False,
        realtimerender=False,
        total_time=5.0,
        dt=0.001,
        stability_tolerance=0.3,
        estimator_lpf_alpha=None,
        initial_angle_spread_deg=0,
        initial_position_spread_m=0.000,
    )


def _default_run_realtime() -> RunParams:
    r = _default_run_offline()
    r.realtimerender = True
    r.total_time = 5.0  # unused when stop policy is INFINITE
    return r


def build_experiment_setup(args: argparse.Namespace) -> ExperimentSetup:
    """Map preset + defaults to real ExperimentSetup (core.sim_types)."""
    plant = PlantParams(
        g=9.81,
        com_length=0.1,
        tau=0.03,
        zeta=0.7,
        num_states=8,
        max_acc=9.81 * 10,
    )
    workspace = WorkspaceParams(
        x_ref=0.0,
        y_ref=-0.0,
        safe_radius=0.068,
    )
    mechanism = MechanismParams(
        O=(128.77, 178.13),
        B=(101.77, 210.13),
        la=175,
        lb=175,
    )

    preset = args.preset
    if preset == "sim":
        hw = _default_hardware()
        run = _default_run_offline()
    elif preset == "vision_real":
        hw = _default_hardware()
        hw.vision_mode = "real_dvs"
        hw.dvs_use_regression = True
        run = _default_run_realtime()
    elif preset == "actuation_real":
        hw = _default_hardware()
        hw.servo = True
        hw.servo_port = None
        run = _default_run_realtime()
    elif preset == "real":
        hw = _default_hardware()
        hw.vision_mode = "real_dvs"
        hw.servo = True
        hw.servo_port = None
        hw.dvs_use_regression = True
        run = _default_run_realtime()
    else:
        raise ValueError(f"Unknown preset: {preset}")

    apply_actuator_to_hardware(hw, resolve_effective_actuator(args))

    variant = BenchmarkVariant(
        controller_type="lqr",
        estimator_type="lpf",
        noise_std=0.01,
        delay_steps=1,
    )

    return ExperimentSetup(
        params=PhysicalParams(
            plant=plant,
            workspace=workspace,
            mechanism=mechanism,
            hardware=hw,
            run=run,
        ),
        camera_params=CameraParams(xr=0.170, yr=0.176),
        default_variant=variant,
    )


def apply_cli_overrides(setup: ExperimentSetup, args: argparse.Namespace) -> None:
    if args.controller:
        setup.default_variant.controller_type = args.controller
    if args.estimator:
        setup.default_variant.estimator_type = normalize_estimator(args.estimator)
    if setup.default_variant.controller_type == "circle":
        radius = setup.params.workspace.safe_radius
        setup.params.workspace.safe_radius = (
            radius if radius is not None else 0.08
        )


def resolve_trials(experiment_type: str, trials_override: int | None) -> int:
    if experiment_type == "single":
        return 1
    return trials_override if trials_override is not None else DEFAULT_TRIALS[experiment_type]


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


def filter_benchmark_variants(args: argparse.Namespace):
    """Optional --controller / --estimator narrow the benchmark sweep."""
    variants = build_default_variants()
    if not args.controller and not args.estimator:
        return variants
    c_f = args.controller
    e_f = normalize_estimator(args.estimator)
    out = []
    for v in variants:
        if c_f and v.controller_type != c_f:
            continue
        if e_f and v.estimator_type != e_f:
            continue
        out.append(v)
    if not out:
        raise ValueError(
            "No benchmark variants match the given --controller/--estimator filters."
        )
    return out


def run(args: argparse.Namespace) -> None:
    if getattr(args, "graph", False):
        run_full_analysis()
        return

    validate_preset_experiment(args)

    setup = build_experiment_setup(args)
    apply_cli_overrides(setup, args)
    if getattr(args, "overrides_json", None):
        try:
            parsed = json.loads(args.overrides_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --overrides-json payload: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--overrides-json must decode to a JSON object/dict")
        try:
            apply_overrides(setup, parsed)
        except OverrideError as exc:
            raise ValueError(f"Invalid override(s): {exc}") from exc

    experiment_type = args.experiment
    trials = resolve_trials(experiment_type, args.trials)
    preset = args.preset

    # sim-only batch experiments
    if experiment_type == "single":
        if preset == "sim":
            engine = SimulationEngine(stop_policy=StopPolicy.FIXED_TIME)
            experiment = SingleExperiment(engine)
        else:
            engine = RealTimeEngine(stop_policy=StopPolicy.INFINITE)
            experiment = RealExperiment(engine)

    elif experiment_type == "montecarlo":
        engine = SimulationEngine(stop_policy=StopPolicy.EARLY_STOP)
        experiment = MonteCarloExperiment(engine, n_trials=trials)

    elif experiment_type == "benchmark":
        engine = SimulationEngine(stop_policy=StopPolicy.EARLY_STOP)
        experiment = BenchmarkExperiment(
            engine,
            variants=filter_benchmark_variants(args),
            n_trials=trials,
        )

    elif experiment_type == "sweep":
        engine = SimulationEngine(stop_policy=StopPolicy.EARLY_STOP)
        experiment = WorkspaceSweepExperiment(
            engine,
            min_diameter_mm=40,
            max_diameter_mm=80,
            n_sizes=20,
            n_trials=trials,
        )

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    result = experiment.run(setup)

    if experiment_type == "benchmark":
        save_benchmark_results(result)
    elif experiment_type in ("single", "montecarlo"):
        print_summary(result)
    elif experiment_type == "sweep":
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run balancer experiments with preset-based CLI (see new_main contract).",
    )
    parser.add_argument(
        "--preset",
        default="sim",
        choices=["sim", "vision_real", "actuation_real", "real"],
        help="Hardware/vision profile",
    )
    parser.add_argument(
        "--experiment",
        default="single",
        choices=["single", "montecarlo", "benchmark", "sweep"],
        help="Which experiment runner to use",
    )
    parser.add_argument(
        "--actuator",
        default=None,
        choices=["sim", "servo", "mock"],
        help="Actuation: sim | servo (/dev/ttyUSB0) | mock (servo path, no port). "
        "Default depends on --preset.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override trial count for montecarlo/benchmark/sweep",
    )
    parser.add_argument(
        "--controller",
        default=None,
        choices=["lqr", "pole", "circle", "null"],
        help="Override default_variant controller",
    )
    parser.add_argument(
        "--estimator",
        default=None,
        choices=["kalman", "lpf", "fde"],
        help="Override default_variant estimator (fde = finite-difference)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Workspace radius (m) for circle controller when --controller circle",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Run offline analysis pipeline (graphing) and exit",
    )
    parser.add_argument(
        "--overrides-json",
        default=None,
        help="Dev override JSON object for dot-path setup fields, e.g. "
        "'{\"params.run.dt\":0.002}'",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
