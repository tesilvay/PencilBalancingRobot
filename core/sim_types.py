from dataclasses import dataclass
import numpy as np

# -----------------------------
# Core State Types
# -----------------------------

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


# -----------------------------
# Vision Types
# -----------------------------

@dataclass
class CameraParams:
    xr: float  # camera 2 x-offset
    yr: float  # camera 1 y-offset


@dataclass
class CameraObservation:
    slope: float
    intercept: float


@dataclass
class CameraPair:
    cam1: CameraObservation
    cam2: CameraObservation


@dataclass
class PoseMeasurement:
    X: float
    Y: float
    alpha_x: float
    alpha_y: float
    
    
# -----------------------------
# Domain-specific parameter groups
# -----------------------------

@dataclass
class PlantParams:
    """Dynamics: gravity, pencil, table, limits."""
    g: float
    com_length: float
    tau: float
    zeta: float
    num_states: int
    max_acc: float | None = None


@dataclass
class WorkspaceParams:
    """Reference position and workspace boundary."""
    x_ref: float
    y_ref: float
    safe_radius: float | None = None


def make_reference_state(workspace: WorkspaceParams) -> SystemState:
    """Build reference state from workspace params."""
    return SystemState(
        x=workspace.x_ref, x_dot=0.0, alpha_x=0.0, alpha_x_dot=0.0,
        y=workspace.y_ref, y_dot=0.0, alpha_y=0.0, alpha_y_dot=0.0
    )


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
    dvs_cam_x_port: str | None = None
    dvs_cam_y_port: str | None = None
    dvs_algo: str = "hough"  # "hough" | "sam"
    dvs_noise_filter_duration_ms: float | None = 30  # None = no filter; > 0 = duration (Sam only)
    dvs_hough_decay: float = 0.95  # Hough only: 0.95 is a good default, 0.9-0.98 is typical, 0.999 is usually too laggy


@dataclass
class RunParams:
    """Simulation/display options."""
    save_video: bool = False
    realtimerender: bool = False
    total_time: float = 5.0
    stability_tolerance: float = 0.05
    estimator_lpf_alpha: float | None = None  # None = LPF default (0.95)


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


@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray

@dataclass
class BenchmarkVariant:
    """One point in the benchmark sweep: controller, estimator, noise, delay."""
    controller_type: str
    estimator_type: str | None
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


@dataclass
class BenchmarkSummary:
    stability_rate: float
    avg_settling_time: float | None
    max_acc: float
    avg_acc: float

@dataclass
class BenchmarkResult:
    params: PhysicalParams
    variant: BenchmarkVariant
    summary: BenchmarkSummary