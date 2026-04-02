from dataclasses import dataclass
import numpy as np


@dataclass
class NullParams:
    """Shared params for classes that take no configuration at init (null / noop / empty)."""


@dataclass
class TimingParams:
    total_time: float
    dt:         float

@dataclass
class PlantParams:
    g: float
    l: float
    tau: float
    zeta: float
    max_acc: float | None = None
    num_states: int
    x_ref: float
    y_ref: float
    safe_radius: float | None = None

@dataclass
class PoleParams:
    plant:  PlantParams   # physical constants live here, no copying
    poles:  list[float]
    
    
@dataclass
class LQRParams:
    plant:          PlantParams   # physical constants live here, no copying
    Q_single_axis:  np.ndarray
    R:              np.ndarray 


@dataclass
class SmoothPoleParams:
    plant:          PlantParams   # physical constants live here, no copying
    timing:         TimingParams
    s_poles:        list[float]
    slew_poles:     float 
    
@dataclass
class SmoothLQRParams:
    plant:          PlantParams   # physical constants live here, no copying
    Q_single_axis:  np.ndarray
    q_u:            float
    r_delta:        float
    
@dataclass
class CircleParams:
    plant:          PlantParams   # physical constants live here, no copying
    timing:         TimingParams
    period_s:       float


# ── Estimators ────────────────────────────────────────────────

@dataclass
class LPFParams:
    alpha: float

@dataclass
class KalmanParams:
    plant:      PlantParams
    timing:     TimingParams
    q_pose_pos: float
    q_pose_ang: float
    q_vel_pos:  float
    q_vel_ang:  float
    r_pose_pos: float
    r_pose_ang: float

@dataclass
class FullKalmanParams:
    plant:      PlantParams
    timing:     TimingParams
    q_pose_pos: float
    q_pose_ang: float
    q_vel_pos:  float
    q_vel_ang:  float
    r_pose_pos: float
    r_vel_pos:  float
    r_pose_ang: float
    r_vel_ang:  float
    lpf_alpha:  float


# ── Vision: Line Algorithms ──────────────────────────────────

@dataclass
class HoughLineParams:
    mixing_factor:    float
    inlier_stddev_px: float
    min_determinant:  float
    max_events:       int | None = None

@dataclass
class SamLineParams:
    min_points: int


# ── Vision: Regression Models ────────────────────────────────

@dataclass
class SimpleRegressionParams:
    calibration_path: str


# ── Vision: Interfaces ───────────────────────────────────────

@dataclass
class SimAnalyticParams:
    noise_std:   float | None
    delay_steps: int

@dataclass
class SimDVSParams:
    dvs_mask_line_y_cam1: int
    dvs_mask_line_y_cam2: int

@dataclass
class RealDVSParams:
    cam1_device:             str | None
    cam2_device:             str | None
    dvs_mask_line_y_cam1:    int
    dvs_mask_line_y_cam2:    int
    noise_filter_duration_ms: float | None = None


# ── Vision: Composite ────────────────────────────────────────

@dataclass
class VisionParams:
    interface: object
    algo:      object
    reg_model: object


# ── Actuators ─────────────────────────────────────────────────

@dataclass
class ServoParams:
    port:      str
    frequency: int


# ── Supervisors ──────────────────────────────────────────────

@dataclass
class DynamicSupervisorParams:
    stable_threshold:  float
    stable_hold_s:     float
    consistent_hold_s: float
    loss_threshold:    float

@dataclass
class StaticSupervisorParams:
    controller_key: str
    estimator_key:  str


# ── System ────────────────────────────────────────────────────

@dataclass
class SystemParams:
    plant:       object
    controllers: dict
    estimators:  dict
    vision:      object
    actuator:    object
    supervisor:  object


# ── Stop Conditions ──────────────────────────────────────────

@dataclass
class FallConditionParams:
    max_angle_deg: float

@dataclass
class StabilizedParams:
    tol_ang_deg: float
    tol_m:       float
    settle_time: float

@dataclass
class MaxStepsConditionParams:
    timing:      TimingParams
    tol_ang_deg: float
    tol_m:       float
    settle_time: float

@dataclass
class AnyStopConditionParams:
    conditions: dict


# ── Visualizers ───────────────────────────────────────────────

@dataclass
class SimDvsVisualizerParams:
    width:  int
    height: int

@dataclass
class RealDvsVisualizerParams:
    width:       int
    height:      int
    mask_y_cam1: int
    mask_y_cam2: int

@dataclass
class OneDvsVisualizerParams:
    cam_index:    int
    width:        int
    height:       int
    surface_gain: float

@dataclass
class SimDvsWorkspaceVisualizerParams:
    width:  int
    height: int

@dataclass
class RealDvsWorkspaceVisualizerParams:
    width:       int
    height:      int
    mask_y_cam1: int
    mask_y_cam2: int

@dataclass
class Visualizer3DParams:
    L:   float
    fps: int


# ── Progress ──────────────────────────────────────────────────

@dataclass
class ProgressParams:
    width: int


# ── Pacing ────────────────────────────────────────────────────

@dataclass
class RealTimePacingParams:
    timing: TimingParams


# ── Scheduler ─────────────────────────────────────────────────

@dataclass
class SchedulerParams:
    timing:             TimingParams
    actuator_frequency: int
    render_frequency:   int


# ── Experiment ────────────────────────────────────────────────

@dataclass
class ExperimentParams:
    system:         object
    logger:         object
    stop_condition: object
    visualizer:     dict
    progress:       object
    pacing:         object
    scheduler:      object

