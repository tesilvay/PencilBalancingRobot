from dataclasses import dataclass, field

import numpy as np

from .base import BasePlant
from src.shared import (
    PlantParams,
    WorkspaceParams,
    State,
    TableAccel,
    ControlInput,
    default_plant,
    default_workspace,
)


@dataclass
class PlacingParams:
    restitution: float
    stick_speed_thresh: float
    slip_damping: float
    plant: PlantParams = field(default_factory=default_plant)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    top_radius: float = 0.0


PLACING_PRESETS = {
    "steady_hands": {
        "top_radius": 0.0,
        "restitution": 0.1,
        "stick_speed_thresh": 0.01,
        "slip_damping": 0.2,
    },
    "shaky": {
        "base": "steady_hands",
        "top_radius": 2.0e-3,
    },
    "angle_only": {
        "base": "steady_hands",
        "top_radius": 0.0,
    },
}


class PlacingPlant(BasePlant):
    """
    Balancer plant with an optional top-point clamp.

    Interpretation:
    - top_radius = 0      -> pencil top is pinned to the initial anchor
    - top_radius > 0      -> pencil top can move inside a disk around the anchor
    - top_radius = large  -> approaches free BalancerPlant behavior, as long as
                             the top never reaches the disk boundary
    """

    def __init__(self, params: PlacingParams):
        p = params.plant
        w = params.workspace

        super().__init__(w, p.max_acc)

        self.g = p.g
        self.l = p.com_length
        self.tau = p.tau
        self.zeta = p.zeta

        self.top_radius = max(float(params.top_radius), 0.0)
        self._top_anchor: np.ndarray | None = None
        
        self.restitution = params.restitution
        self.stick_speed_thresh = params.stick_speed_thresh
        self.slip_damping = params.slip_damping

    def reset(self) -> None:
        self._top_anchor = None

    def set_top_radius(self, radius: float) -> None:
        self.top_radius = max(float(radius), 0.0)

    def _ensure_top_anchor(self, state_x: State) -> np.ndarray:
        if self._top_anchor is None:
            self._top_anchor = np.array(
                [
                    float(state_x.px + self.l * state_x.ax),
                    float(state_x.py + self.l * state_x.ay),
                ],
                dtype=float,
            )
        return self._top_anchor

    def _constrain_top(
        self,
        top_pos: np.ndarray,
        top_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        anchor = self._top_anchor
        if anchor is None:
            return top_pos, top_vel

        delta = top_pos - anchor
        dist = float(np.linalg.norm(delta))

        if self.top_radius <= 0.0:
            return anchor.copy(), np.zeros(2, dtype=float)

        if dist <= self.top_radius:
            return top_pos, top_vel

        normal = delta / dist
        top_pos = anchor + normal * self.top_radius

        radial_speed = float(np.dot(top_vel, normal))
        v_normal = radial_speed * normal
        v_tangent = top_vel - v_normal
        tangent_speed = float(np.linalg.norm(v_tangent))

        # prevent outward motion, allow slight bounce inward
        if radial_speed > 0.0:
            v_normal = -self.restitution * v_normal

        # stick-slip friction on the boundary
        if tangent_speed < self.stick_speed_thresh:
            v_tangent = np.zeros(2, dtype=float)
        else:
            v_tangent = self.slip_damping * v_tangent

        top_vel = v_normal + v_tangent
        return top_pos, top_vel

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        px = state_x.px
        vx = state_x.vx
        ax = state_x.ax
        wx = state_x.wx

        py = state_x.py
        vy = state_x.vy
        ay = state_x.ay
        wy = state_x.wy

        px_cmd, py_cmd = u_cmd.px_cmd, u_cmd.py_cmd

        self._ensure_top_anchor(state_x)

        # Free balancer dynamics
        vx_dot = (1.0 / self.tau**2) * (px_cmd - px) - (2.0 * self.zeta / self.tau) * vx
        vy_dot = (1.0 / self.tau**2) * (py_cmd - py) - (2.0 * self.zeta / self.tau) * vy
        vx_dot, vy_dot = self.clamp_acceleration(vx_dot, vy_dot)

        wx_dot = (self.g / self.l) * ax - (1.0 / self.l) * vx_dot
        wy_dot = (self.g / self.l) * ay - (1.0 / self.l) * vy_dot

        # Integrate base motion
        vx += vx_dot * dt
        px += vx * dt

        vy += vy_dot * dt
        py += vy * dt

        px, vx, py, vy = self.apply_workspace_limits(px, vx, py, vy)

        # Integrate free angular motion
        wx += wx_dot * dt
        wy += wy_dot * dt

        ax_free = ax + wx * dt
        ay_free = ay + wy * dt

        # Convert free state to top point
        top_pos = np.array(
            [
                float(px + self.l * ax_free),
                float(py + self.l * ay_free),
            ],
            dtype=float,
        )

        top_vel = np.array(
            [
                float(vx + self.l * wx),
                float(vy + self.l * wy),
            ],
            dtype=float,
        )

        # Apply clamp
        top_pos, top_vel = self._constrain_top(top_pos, top_vel)

        # Convert constrained top back to angle/angular velocity
        ax = float((top_pos[0] - px) / self.l)
        ay = float((top_pos[1] - py) / self.l)
        wx = float((top_vel[0] - vx) / self.l)
        wy = float((top_vel[1] - vy) / self.l)

        ax = float(np.clip(ax, -np.pi / 2, np.pi / 2))
        ay = float(np.clip(ay, -np.pi / 2, np.pi / 2))

        return (
            State(
                px=px,
                vx=vx,
                ax=ax,
                wx=wx,
                py=py,
                vy=vy,
                ay=ay,
                wy=wy,
            ),
            TableAccel(x_ddot=vx_dot, y_ddot=vy_dot),
        )