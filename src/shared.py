from dataclasses import dataclass, field
import numpy as np


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
class SystemState:
    px: float
    vx: float
    ax: float
    wx: float
    py: float
    vy: float
    ay: float
    wy: float

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.vx, self.ax, self.wx,
            self.py, self.vy, self.ay, self.wy,
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
        return np.array([self.x_ddot, self.y_ddot])


@dataclass
class PoseMeasurement:
    X: float
    Y: float
    alpha_x: float
    alpha_y: float


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

def _build_camera_params(params: "CameraParams") -> "CameraParams":
    """Identity builder so build_from_registry can construct CameraParams."""
    return params

CAMERA_PRESETS_REGISTRY = {
    "default": Spec(_build_camera_params, CameraParams, CAMERA_PRESETS)
}


@dataclass
class CameraObservation:
    slope: float
    intercept: float


@dataclass
class CameraPair:
    cam1: CameraObservation
    cam2: CameraObservation


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_reference_state(workspace: WorkspaceParams) -> SystemState:
    """Build reference state from workspace params."""
    return SystemState(
        px=workspace.x_ref, vx=0.0, ax=0.0, wx=0.0,
        py=workspace.y_ref, vy=0.0, ay=0.0, wy=0.0,
    )


def clamp_table_command_to_workspace(
    cmd: TableCommand, workspace: WorkspaceParams
) -> TableCommand:
    """Clamp a table command to the circular workspace boundary."""
    x_des, y_des = cmd.x_des, cmd.y_des
    safe_radius = workspace.safe_radius

    if safe_radius is None:
        return TableCommand(x_des, y_des)

    dx = x_des - workspace.x_ref
    dy = y_des - workspace.y_ref
    dist = float(np.sqrt(dx * dx + dy * dy))

    if dist > safe_radius and dist > 0:
        scale = safe_radius / dist
        x_des = workspace.x_ref + dx * scale
        y_des = workspace.y_ref + dy * scale

    return TableCommand(x_des, y_des)


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
        elif isinstance(v, dict) and sub_registry:
            resolved[k] = {
                name: build_from_registry(sub_registry, s)
                for name, s in v.items()
            }
        else:
            resolved[k] = v

    return spec.cls(spec.Params(**resolved))
