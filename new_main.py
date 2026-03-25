from dataclasses import dataclass, field

from core.sim_types import (
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


# ---------- DOMAIN (unchanged physics) ----------
@dataclass
class PhysicalParams:
    plant: any
    workspace: any
    mechanism: any

@dataclass
class PlantParams:
    """Dynamics: gravity, pencil, table, limits."""
    g: float = 9.81
    com_length: float = 0.1
    tau: float = 0.04
    zeta: float = 0.7
    num_states: int = 8
    max_acc: float | None = None

@dataclass
class WorkspaceParams:
    """Reference position and workspace boundary."""
    x_ref: float = 0.0
    y_ref: float = 0.0
    safe_radius: float | None = None

@dataclass
class MechanismParams:
    """Five-bar geometry (mm)."""
    O: tuple[float, float] = (128.77, 178.13)
    B: tuple[float, float] = (101.77, 210.13)
    la: float = 175.0
    lb: float = 175.0


@dataclass
class CameraParams:
    xr: float = 0.170 # camera 2 x-offset
    yr: float = 0.176 # camera 1 y-offset


@dataclass
class BenchmarkVariant:
    """One point in the benchmark sweep: controller, estimator, noise, delay."""
    controller_type: str  # "lqr" | "pole" | "circle"
    estimator_type: str   # "kalman" | "lpf" | "fde"
    noise_std: float
    delay_steps: int


@dataclass
class SystemConfig:
    """System, what is real and what is sim"""
    plant: str        # "sim" | "real"
    sensor: str       # "sim_analytic" | "sim_dvs" | "real_dvs"
    actuator: str     # "sim" | "servo" | "mock"


@dataclass
class RuntimeConfig:
    """How it runs"""
    mode: str = "offline"          # "offline" | "realtime"
    dt: float = 0.001
    duration: float | None = 5.0   # None = infinite
    render: bool = False


@dataclass
class ExperimentConfig:
    type: str = "single"   # "single" | "montecarlo" | "benchmark" | "sweep"
    trials: int = 1


@dataclass
class ExperimentSetup:
    domain: PhysicalParams
    system: SystemConfig
    runtime: RuntimeConfig
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    variant: BenchmarkVariant
    camera_params: CameraParams
    

@dataclass
class HoughTrackerParams:
    """Tuneables for the original Java-style recursive Hough tracker."""
    mixing_factor: float = 0.02  # Hough only: inlier adaptation rate per event; 0.01-0.05 is a good starting range.
    inlier_stddev_px: float = 4.0  # Hough only: Gaussian inlier width in pixels; 3-6 px is typical, larger follows faster motion but admits more noise.
    min_determinant: float = 1e-6  # Hough only: reject unstable solves when the quadratic becomes degenerate; usually leave near 1e-6.


@dataclass
class VisionConfig:
    mode: str = "sim_analytic" # sim_analytic | sim_dvs | real_dvs
    dvs_algo: str = "hough" # "hough" | "sam"
    use_regression: bool = False
    hough: HoughTrackerParams = field(default_factory=HoughTrackerParams)
    sam_filter_ms: int | None = 5

@dataclass
class ActuatorConfig:
    type: str = "sim"  # "sim" | "servo" | "mock"
    port: str | None = None
    frequency: int = 250


from configs import *
import argparse


def make_preset(args):
    name = args.preset
    

    # ----- shared domain -----
    domain = PhysicalParams(
        plant=PlantParams(max_acc=9.81 * 3),
        workspace=WorkspaceParams(safe_radius=0.108),
        mechanism=MechanismParams(),
    )

    camera = CameraParams()

    variant = BenchmarkVariant(
        controller_type="lqr",  # "lqr" | "pole" | "circle"
        estimator_type="kalman", # "kalman" | "lpf" | "fde"
        noise_std=0.01,
        delay_steps=0,
    )

    experiment = ExperimentConfig(type=args.experiment)
    
    # ----- presets -----
    if name == "sim":
        system = SystemConfig(
            plant="sim",
            sensor="sim_analytic",
            actuator="sim",
        )
        runtime = RuntimeConfig(
            mode="offline",
            duration=5.0,
            render=False,
        )

    elif name == "vision_real":  # real DVS, simulated actuation
        system = SystemConfig(
            plant="sim",
            sensor="real_dvs",
            actuator="sim",
        )
        runtime = RuntimeConfig(
            mode="realtime",
            duration=None,
            render=True,
        )

    elif name == "actuation_real":  # real servo, simulated vision
        system = SystemConfig(
            plant="sim",
            sensor="sim_analytic",
            actuator="servo",
        )
        runtime = RuntimeConfig(
            mode="realtime",
            duration=None,
            render=True,
        )

    elif name == "real":  # fully real system
        system = SystemConfig(
            plant="real",
            sensor="real_dvs",
            actuator="servo",
        )
        runtime = RuntimeConfig(
            mode="realtime",
            duration=None,
            render=True,
        )

    else:
        raise ValueError(f"Unknown preset: {name}")

    return ExperimentSetup(
        domain=domain,
        system=system,
        runtime=runtime,
        variant=variant,
        experiment=experiment,
        camera_params=camera,
    ) 


DEFAULT_TRIALS = {
    "single": 1,
    "montecarlo": 200,
    "benchmark": 100,
    "sweep": 100,   # maybe smaller, depends on cost
}

VALID = {
        "sim": {"single", "montecarlo", "benchmark", "sweep"},
        "vision_real": {"single"},
        "actuation_real": {"single"},
        "real": {"single"},
    }

def validate_mode_and_experiment(args):
    
    if args.experiment not in VALID[args.preset]:
        raise ValueError(
            f"Experiment '{args.experiment}' not allowed in preset '{args.preset}'"
        )

def make_experiment(setup, trials_override=None):

    exp_type = setup.experiment.type

    # resolve trials
    if exp_type == "single":
        setup.experiment.trials = 1

    else:
        setup.experiment.trials = trials_override or DEFAULT_TRIALS[exp_type]

    # build experiment
    if exp_type == "single":
        return SingleExperiment(setup)

    elif exp_type == "montecarlo":
        return MonteCarloExperiment(setup)

    elif exp_type == "benchmark":
        return BenchmarkExperiment(setup)

    elif exp_type == "sweep":
        return WorkspaceSweepExperiment(setup)

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", default="sim",
                        choices=["sim", "vision_real", "actuation_real", "real"])

    parser.add_argument("--experiment", default="single",
                        choices=["single", "montecarlo", "benchmark", "sweep"])

    parser.add_argument("--trials", type=int, default=None,
                        help="Override number of trials (for montecarlo/benchmark/sweep)")

    parser.add_argument("--controller", default=None,
                        choices=["lqr", "pole", "circle"])
    
    parser.add_argument("--estimator", default=None,
                        choices=["kalman", "lpf", "fde"])

    parser.add_argument("--radius", type=float, default=None)

    args = parser.parse_args()
    
    # ---- validate mode vs experiment ----
    validate_mode_and_experiment(args)

    # ---- setup ----
    setup = make_preset(args)

    # ---- controller override ----
    if args.controller:
        setup.variant.controller_type = args.controller
        
    if args.estimator:
        setup.variant.estimator_type = args.estimator

    # optional circle params
    if args.controller == "circle":
        setup.domain.workspace.safe_radius = args.radius or 0.1

    # ---- build experiment ----
    experiment = make_experiment(setup, trials_override=args.trials)

    # ---- run ----
    experiment.run()


if __name__ == "__main__":
    main()
