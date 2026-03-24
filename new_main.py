from dataclasses import dataclass

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


# ---------- DOMAIN (unchanged physics) ----------
@dataclass
class PhysicalParams:
    plant: any
    workspace: any
    mechanism: any


# ---------- SYSTEM (what is real vs sim) ----------
@dataclass
class SystemConfig:
    plant: str        # "sim" | "real"
    sensor: str       # "sim_analytic" | "sim_dvs" | "real_dvs"
    actuator: str     # "sim" | "servo"


# ---------- RUNTIME (how it runs) ----------
@dataclass
class RuntimeConfig:
    mode: str         # "offline" | "realtime"
    dt: float
    duration: float | None   # None = infinite
    render: bool


# ---------- EXPERIMENT ----------
@dataclass
class ExperimentConfig:
    type: str         # "single" | "montecarlo" | "benchmark" | "sweep"
    trials: int = 1


# ---------- FULL SETUP ----------
@dataclass
class ExperimentSetup:
    domain: PhysicalParams
    system: SystemConfig
    runtime: RuntimeConfig
    experiment: ExperimentConfig
    variant: any
    camera_params: any


from configs import *

def make_preset(name: str):

    # ----- shared domain (your existing params) -----
    domain = PhysicalParams(
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
    )

    camera = CameraParams(xr=0.170, yr=0.176)

    variant = BenchmarkVariant(
        controller_type="lqr",
        estimator_type="kalman",
        noise_std=0.01,
        delay_steps=1,
    )

    # ----- modes -----
    if name == "sim":
        system = SystemConfig(
            plant="sim",
            sensor="sim_analytic",
            actuator="sim",
        )
        runtime = RuntimeConfig(
            mode="offline",
            dt=0.001,
            duration=5.0,
            render=False,
        )

    elif name == "hybrid":
        system = SystemConfig(
            plant="sim",
            sensor="real_dvs",
            actuator="servo",
        )
        runtime = RuntimeConfig(
            mode="realtime",
            dt=0.001,
            duration=None,
            render=True,
        )

    elif name == "real":
        system = SystemConfig(
            plant="real",
            sensor="real_dvs",
            actuator="servo",
        )
        runtime = RuntimeConfig(
            mode="realtime",
            dt=0.001,
            duration=None,
            render=True,
        )

    else:
        raise ValueError(f"Unknown preset: {name}")

    return ExperimentSetup(
        domain=domain,
        system=system,
        runtime=runtime,
        experiment=ExperimentConfig(type="single"),
        variant=variant,
        camera_params=camera,
    )