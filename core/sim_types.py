from dataclasses import dataclass, field
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


@dataclass
class HoughQuadraticState:
    """
    Quadratic coefficients for the continuous Hough objective
    J(m, b) = a*m^2 + cross_mb*m*b + c*b^2 + linear_m*m + linear_b*b.
    """
    quadratic_m2: float = 0.0
    cross_mb: float = 0.0
    quadratic_b2: float = 0.0
    linear_m: float = 0.0
    linear_b: float = 0.0


@dataclass
class HoughTrackerParams:
    """Tuneables for the original Java-style recursive Hough tracker."""
    mixing_factor: float = 0.02  # Hough only: inlier adaptation rate per event; 0.01-0.05 is a good starting range.
    inlier_stddev_px: float = 4.0  # Hough only: Gaussian inlier width in pixels; 3-6 px is typical, larger follows faster motion but admits more noise.
    min_determinant: float = 1e-6  # Hough only: reject unstable solves when the quadratic becomes degenerate; usually leave near 1e-6.


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
    
    vision_mode: str = "sim_analytic" # "real_dvs" | "sim_dvs" | "sim_analytic"
    
    dvs_cam_x_port: str | None = None
    dvs_cam_y_port: str | None = None
    dvs_mask_line_y_cam1: int = 160
    dvs_mask_line_y_cam2: int = 190
    servo_frequency: int = 250  # Hz; command update rate to servos
    dvs_algo: str = "hough"  # "hough" | "sam"
    sam_filter_ms: float | None = 30  # None = no filter; > 0 = duration (Sam OLS only)
    dvs_hough: HoughTrackerParams = field(default_factory=HoughTrackerParams)  # Hough only: ignored when dvs_algo="sam"
    dvs_use_regression: bool = False  # When true, use learned multivariate regression for pose instead of pure analytic reconstruct


@dataclass
class RunParams:
    """Simulation/display options."""
    save_video: bool = False
    realtimerender: bool = False
    total_time: float = 5.0
    dt: float = 0.001
    stability_tolerance: float = 0.05
    estimator_lpf_alpha: float | None = None  # None = LPF default (0.95)
    # Monte Carlo initial state: ± spread for angle (degrees) and position (meters)
    initial_angle_spread_deg: float = 11.46  # 0.2 rad ≈ 11.46°
    initial_position_spread_m: float = 0.050  # ±50 mm


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

@dataclass
class TerminalInfo:
    stabilized: bool
    settling_time: float | None

@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray
    mech_history: np.ndarray | None = None
    cmd_history: np.ndarray | None = None 
    terminal: TerminalInfo | None = None

@dataclass
class StopPolicy:
    FIXED_TIME = "fixed_time"
    EARLY_STOP = "early_stop"
    INFINITE = "infinite"