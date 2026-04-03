from dataclasses import dataclass, field
import numpy as np

# PLANT
@dataclass
class PlantParams:
    g: float
    l: float
    tau: float
    zeta: float
    num_states: int
    x_ref: float
    y_ref: float
    max_acc: float | None = None
    safe_radius: float | None = None

PLANT_PRESETS = {
    "default": {
        "g": 9.81, "l": 0.15, "tau": 0.03, "zeta": 0.8,
        "max_acc": 9.81 * 3, "num_states": 8,
        "x_ref": 0.0, "y_ref": 0.0, "safe_radius": 68e-3,
    }
}

# WORKSPACE
@dataclass
class WorkspaceParams:
    """Reference position and workspace boundary."""
    x_ref: float
    y_ref: float
    safe_radius: float | None = None

# TIMING
@dataclass
class TimingParams:
    total_time: float = 5.0
    dt:         float = 4e-3

TIMING_PRESETS = {
    "default": {"total_time": 5.0, "dt": 4e-3},
    "long":    {"total_time": 30.0, "dt": 4e-3},
}

# NULL
@dataclass
class NullParams:
    pass

NULL_PRESETS = {"default": {}}

# SPEC
@dataclass
class Spec:
    cls:        type
    Params:     type
    Presets:    dict
    registries: dict | None = None
    sim_only:   bool | None = None

# LOOP DATACLASSES
@dataclass
class SystemState:
    x: float
    x_dot: float
    alpha_x: float
    alpha_x_dot: float
    y: float
    y_dot: float
    alpha_y: float
    alpha_y_dot: float

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.x,
            self.x_dot,
            self.alpha_x,
            self.alpha_x_dot,
            self.y,
            self.y_dot,
            self.alpha_y,
            self.alpha_y_dot
        ])


@dataclass
class TableCommand:
    x_des: float
    y_des: float

@dataclass
class TableAccel:
    x_ddot: float
    y_ddot: float
    
    def as_vector(self) -> np.ndarray:
        return np.array([
            self.x_ddot,
            self.y_ddot
        ])

@dataclass
class PoseMeasurement:
    X: float
    Y: float
    alpha_x: float
    alpha_y: float


# VISION TYPES
@dataclass
class CameraParams:
    xr: float  # camera 2 x-offset
    yr: float  # camera 1 y-offset
    y_mask_line_1:   float
    y_mask_line_2:   float
    DAVIS346_WIDTH:  float
    DAVIS346_HEIGHT: float

CAMERA_PRESETS = {
    "default": {
        "xr": 170, 
        "yr": 360, 
        "y_mask_line_1":160, 
        "y_mask_line_2":190,
        "DAVIS346_WIDTH":  346,
        "DAVIS346_HEIGHT": 260,
    },
}

@dataclass
class CameraObservation:
    slope: float
    intercept: float

@dataclass
class CameraPair:
    cam1: CameraObservation
    cam2: CameraObservation


def make_reference_state(workspace: WorkspaceParams) -> SystemState:
    """Build reference state from workspace params."""
    return SystemState(
        x=workspace.x_ref, x_dot=0.0, alpha_x=0.0, alpha_x_dot=0.0,
        y=workspace.y_ref, y_dot=0.0, alpha_y=0.0, alpha_y_dot=0.0
    )


# HARDWARE / MECHANISM
@dataclass
class MechanismParams:
    """Five-bar geometry (mm)."""
    O: tuple[float, float]
    B: tuple[float, float]
    la: float
    lb: float

@dataclass
class HardwareParams:
    """Real hardware flags and ports."""
    servo: bool = False
    servo_port: str | None = None
    dvs_cam: bool = False
    vision_mode: str = "sim_analytic"
    dvs_cam_x_port: str | None = None
    dvs_cam_y_port: str | None = None
    dvs_mask_line_y_cam1: int = 160
    dvs_mask_line_y_cam2: int = 190
    servo_frequency: int = 250
    dvs_algo: str = "hough"
    sam_filter_ms: float | None = 30
    dvs_hough: HoughTrackerParams = field(default_factory=HoughTrackerParams)
    dvs_use_regression: bool = False

@dataclass
class RunParams:
    """Simulation/display options."""
    save_video: bool = False
    realtimerender: bool = False
    total_time: float = 5.0
    dt: float = 0.001
    stability_tolerance_deg: float = 10
    stability_tolerance_m: float = 10e-3
    settle_time: float = 0.5
    estimator_lpf_alpha: float | None = None
    initial_angle_spread_deg: float = 11.46
    initial_position_spread_m: float = 0.050
    initial_linear_velocity_spread_mps: float = 0
    initial_angular_velocity_spread_degps: float = 0
    estimator_diagnostics_enabled: bool = False
    estimator_diagnostics_terminal_hz: float = 2.0
    estimator_diagnostics_terminal: bool = True
    estimator_diagnostics_history: int = 0

@dataclass
class PhysicalParams:
    """Composition of all physical/experiment parameters."""
    plant: PlantParams
    workspace: WorkspaceParams
    mechanism: MechanismParams | None = None
    hardware: HardwareParams | None = None
    run: RunParams | None = None

    def __post_init__(self):
        if self.hardware is None:
            self.hardware = HardwareParams()
        if self.run is None:
            self.run = RunParams()


# BENCHMARK / EXPERIMENT
@dataclass
class BenchmarkVariant:
    """One point in the benchmark sweep: controller, estimator, noise, delay."""
    controller_type: str
    estimator_type: str
    noise_std: float
    delay_steps: int

@dataclass
class ExperimentSetup:
    """Bundled experiment configuration: params, cameras, and default algorithm variant."""
    params: PhysicalParams
    camera_params: CameraParams
    default_variant: BenchmarkVariant

@dataclass
class TrialMetrics:
    stabilized: bool
    settling_time: float | None
    max_acc: float
    avg_state_est_err: np.ndarray | None = None

@dataclass
class BenchmarkSummary:
    stability_rate: float
    avg_settling_time: float | None
    max_acc: float
    avg_acc: float
    avg_state_est_err: np.ndarray | None = None

@dataclass
class BenchmarkResult:
    params: PhysicalParams
    variant: BenchmarkVariant
    summary: BenchmarkSummary

@dataclass
class TerminalInfo:
    stabilized: bool
    settling_time: float | None

@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray
    mech_history: np.ndarray | None = None
    state_est_err_history: np.ndarray | None = None
    cmd_history: np.ndarray | None = None
    terminal: TerminalInfo | None = None

@dataclass
class StopPolicy:
    FIXED_TIME = "fixed_time"
    EARLY_STOP = "early_stop"
    INFINITE = "infinite"


# BUILDER FUNCS
def resolve_preset(presets, name):
    p = presets[name]
    if "base" in p:
        base = resolve_preset(presets, p["base"])
        return {**base, **{k: v for k, v in p.items() if k != "base"}}
    return p

def build_from_registry(registry, spec_string):
    type_, preset = spec_string.split(":")
    try:
        spec = registry[type_]
    except KeyError:
        raise ValueError(f"Unknown type: {type_}")

    raw = resolve_preset(spec.Presets, preset)

    resolved = {}
    for k, v in raw.items():
        sub_registry = (spec.registries or {}).get(k)
        if isinstance(v, str) and ":" in v:
            resolved[k] = build_from_registry(sub_registry, v)
        elif isinstance(v, dict) and sub_registry:
            resolved[k] = {name: build_from_registry(sub_registry, s) for name, s in v.items()}
        else:
            resolved[k] = v

    return spec.cls(spec.Params(**resolved))


