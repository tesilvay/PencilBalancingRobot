from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from numpy import rad2deg, deg2rad


# ── Plant ─────────────────────────────────────────────────────────────────────

@dataclass
class PlantParams:
    """Pure dynamics: no workspace geometry."""
    g:          float
    com_length: float
    tau:        float
    zeta:       float
    num_states: int
    max_acc:    float | None = None

PLANT_PRESETS = {
    "default": {
        "g": 9.81, "com_length": 0.15, "tau": 0.03, "zeta": 0.8,
        "max_acc": 9.81 * 3, "num_states": 8,
    }
}

def default_plant() -> PlantParams:
    return PlantParams(**PLANT_PRESETS["default"])


# ── Workspace ─────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceParams:
    """Workspace geometry — separate concern from plant dynamics."""
    x_ref:       float
    y_ref:       float
    safe_radius: float | None = None

WORKSPACE_PRESETS = {
    "default": {"x_ref": 0.0, "y_ref": 0.0, "safe_radius": 68e-3}
}

def default_workspace() -> WorkspaceParams:
    return WorkspaceParams(**WORKSPACE_PRESETS["default"])


# ── Timing ────────────────────────────────────────────────────────────────────

@dataclass
class TimingParams:
    total_time: float = 5.0
    dt:         float = 4e-3

TIMING_PRESETS = {
    "default": {"total_time": 5.0, "dt": 4e-3},
    "long":    {"total_time": 30.0, "dt": 4e-3},
}

def default_timing() -> TimingParams:
    return TimingParams(**TIMING_PRESETS["default"])

# ── Initial Conditions ────────────────────────────────────────────────────────

@dataclass
class InitConditionsSpread:
    pos_m:   float
    ang_deg: float
    vel_mps: float
    w_degps: float
    
INIT_CONDITIONS_SPREAD_PRESETS = {
    "easy": {
        "pos_m":   0, 
        "ang_deg": 2,
        "vel_mps": 0,
        "w_degps": 0,
    },
    "angle": {
        "base": "easy",
        "ang_deg": 8,
    },
    "real": {
        "base": "easy",
        "pos_m": 10e3, 
        "ang_deg": 5,
    },
    "hard": {
        "pos_m": 30e-3, 
        "ang_deg": 8,
        "vel_mps": 5e-3,
        "w_degps": 4,
    },
}

def default_spread() -> InitConditionsSpread:
    return InitConditionsSpread(**INIT_CONDITIONS_SPREAD_PRESETS["easy"])
    
# ── Null ──────────────────────────────────────────────────────────────────────

@dataclass
class NullParams:
    pass

NULL_PRESETS = {"default": {}}


# ── Spec ──────────────────────────────────────────────────────────────────────

@dataclass
class Spec:
    cls:        type
    Params:     type
    Presets:    dict
    registries: dict | None = None
    sim_only:   bool | None = None


# ── Loop / actuation types ────────────────────────────────────────────────────

@dataclass
class State:
    px: float
    vx: float
    ax: float
    wx: float
    py: float
    vy: float
    ay: float
    wy: float
    
    @classmethod
    def from_iterable(cls, data: Sequence[float]) -> "State":
        if len(data) != 8:
            raise ValueError(f"Expected 8 values, got {len(data)}")

        return cls(*map(float, data))

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.vx, self.ax, self.wx,
            self.py, self.vy, self.ay, self.wy,
        ])
    
    def state_str(self):
        return (
            f"px={self.px*1000:+.2f} mm, vx={self.vx*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(self.ax):+.2f}°, wx={np.rad2deg(self.wx):+.2f}°/s | "
            f"py={self.py*1000:+.2f} mm, vy={self.vy*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(self.ay):+.2f}°, wy={np.rad2deg(self.wy):+.2f}°/s"
        )
    
    def print_state(self):
        print(f"x:   {self.state_str()}")

    def print_est(self):
        print(f"x_hat:   {self.state_str()}")
    
    def print_vel(self):
        print(
            f"x_vel:   "
            f"vx={self.vx*1000:+.2f} mm/s, "
            f"wx={np.rad2deg(self.wx):+.2f}°/s | "
            f"vy={self.vy*1000:+.2f} mm/s, "
            f"wy={np.rad2deg(self.wy):+.2f}°/s"
        )


@dataclass
class ControlInput:
    px_cmd: float
    py_cmd: float


@dataclass
class TableAccel:
    x_ddot: float
    y_ddot: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x_ddot, self.y_ddot])


@dataclass
class Measurement:
    px: float
    py: float
    ax: float
    ay: float
    
    def as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.ax, self.py, self.ay,
        ])
    
    def print_y_meas(self):
        print(
            f"y:   "
            f"px={self.px*1000:+.2f} mm, "
            f"ax={np.rad2deg(self.ax):+.2f}° | "
            f"py={self.py*1000:+.2f} mm, "
            f"ay={np.rad2deg(self.ay):+.2f}°"
        )


# ── Camera / vision types ─────────────────────────────────────────────────────

@dataclass
class CameraParams:
    xr: float
    yr: float
    y_mask_line_1:   float
    y_mask_line_2:   float
    DAVIS346_WIDTH:  float
    DAVIS346_HEIGHT: float

CAMERA_PRESETS = {
    "default": {
        "xr": 170,
        "yr": 360,
        "y_mask_line_1":   160,
        "y_mask_line_2":   190,
        "DAVIS346_WIDTH":  346,
        "DAVIS346_HEIGHT": 260,
    },
}


def default_camera_params() -> CameraParams:
    return CameraParams(**CAMERA_PRESETS["default"])




@dataclass
class CameraObservation:
    slope: float
    intercept: float


@dataclass
class CameraPair:
    cam1: CameraObservation
    cam2: CameraObservation

    def unpack(self) -> tuple[float, float, float, float]:
        return (
            self.cam1.intercept, 
            self.cam1.slope, 
            self.cam2.intercept, 
            self.cam2.slope)
    


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_reference_state(workspace: WorkspaceParams) -> State:
    """Build reference state from workspace params."""
    return State(
        px=workspace.x_ref, vx=0.0, ax=0.0, wx=0.0,
        py=workspace.y_ref, vy=0.0, ay=0.0, wy=0.0,
    )


def clamp_control_input_to_workspace(
    cmd: ControlInput, workspace: WorkspaceParams
) -> ControlInput:
    """Clamp a control input to the circular workspace boundary."""
    px_cmd, py_cmd = cmd.px_cmd, cmd.py_cmd
    safe_radius = workspace.safe_radius

    if safe_radius is None:
        return ControlInput(px_cmd, py_cmd)

    dx = px_cmd - workspace.x_ref
    dy = py_cmd - workspace.y_ref
    dist = float(np.sqrt(dx * dx + dy * dy))

    if dist > safe_radius and dist > 0:
        scale = safe_radius / dist
        px_cmd = workspace.x_ref + dx * scale
        py_cmd = workspace.y_ref + dy * scale

    return ControlInput(px_cmd, py_cmd)


# ── Registry builders ─────────────────────────────────────────────────────────

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
        raise ValueError(f"Unknown type: {type_!r} — available: {list(registry)}")

    raw = resolve_preset(spec.Presets, preset)

    resolved = {}
    for k, v in raw.items():
        sub_registry = (spec.registries or {}).get(k)
        if isinstance(v, str) and ":" in v:
            resolved[k] = build_from_registry(sub_registry, v)
        elif isinstance(v, list) and sub_registry:
            # Homogeneous collaborators (e.g. controllers): order is part of the
            # contract — supervisors pick active instances by index.
            resolved[k] = [build_from_registry(sub_registry, s) for s in v]
        else:
            resolved[k] = v

    return spec.cls(spec.Params(**resolved))
