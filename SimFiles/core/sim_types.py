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
# Simulation Result
# -----------------------------

@dataclass
class PhysicalParams:
    g: float
    com_length: float
    tau: float
    zeta: float
    num_states: int
    max_acc: float | None = None
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    # mech params
    O: tuple[float, float] | None = None
    B: tuple[float, float] | None = None
    la: float | None = None
    lb: float | None = None

@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray

@dataclass
class ExperimentConfig:
    controller_type: str
    estimator_type: str | None
    noise_std: float
    delay_steps: int

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
    config: ExperimentConfig
    summary: BenchmarkSummary