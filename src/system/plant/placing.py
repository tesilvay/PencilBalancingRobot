from dataclasses import dataclass, field

from .base import BasePlant

import numpy as np

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
    """Balancer-like plant with the pencil top constrained by a fingertip ring."""

    plant: PlantParams = field(default_factory=default_plant)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    top_radius: float = 0.0


PLACING_PRESETS = {
    "steady_hands": {
        "top_radius": 0.0,
    },
    "shaky": {
        "top_radius": 2.0e-3,
    },
    "angle_only": {
        "top_radius": 0.0,
    },
}


class PlacingPlant(BasePlant):
    """Simulate a held pencil by constraining the pencil top instead of letting it fall freely."""

    def __init__(self, params: PlacingParams):
        p = params.plant
        w = params.workspace
        
        super().__init__(w, p.max_acc)

        self.g = p.g
        self.l = p.com_length
        self.tau = p.tau
        self.zeta = p.zeta
        self.top_radius = max(float(params.top_radius), 0.0)

        self._tx = float(w.x_ref)
        self._ty = float(w.y_ref)
        self._tvx = 0.0
        self._tvy = 0.0
        self._top_anchor: np.ndarray | None = None

    def reset(self) -> None:
        self._tx = float(self.x_ref)
        self._ty = float(self.y_ref)
        self._tvx = 0.0
        self._tvy = 0.0
        self._top_anchor = None

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

    def _constrain_top(self, top_pos: np.ndarray, top_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        outward_speed = float(np.dot(top_vel, normal))
        if outward_speed > 0.0:
            top_vel = top_vel - outward_speed * normal
        return top_pos, top_vel

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        px_cmd, py_cmd = u_cmd.px_cmd, u_cmd.py_cmd

        tvx_dot = (1 / self.tau**2) * (px_cmd - self._tx) - (2 * self.zeta / self.tau) * self._tvx
        tvy_dot = (1 / self.tau**2) * (py_cmd - self._ty) - (2 * self.zeta / self.tau) * self._tvy
        tvx_dot, tvy_dot = self.clamp_acceleration(tvx_dot, tvy_dot)

        self._tvx += tvx_dot * dt
        self._tx += self._tvx * dt
        self._tvy += tvy_dot * dt
        self._ty += self._tvy * dt

        self._tx, self._tvx, self._ty, self._tvy = self.apply_workspace_limits(
            self._tx, self._tvx, self._ty, self._tvy
        )

        px = state_x.px
        vx = state_x.vx
        ax = state_x.ax
        wx = state_x.wx

        py = state_x.py
        vy = state_x.vy
        ay = state_x.ay
        wy = state_x.wy

        self._ensure_top_anchor(state_x)

        vx_dot = tvx_dot
        vy_dot = tvy_dot

        wx_dot = (self.g / self.l) * ax - (1 / self.l) * vx_dot
        wy_dot = (self.g / self.l) * ay - (1 / self.l) * vy_dot

        vx += vx_dot * dt
        px += vx * dt
        vy += vy_dot * dt
        py += vy * dt

        px, vx, py, vy = self.apply_workspace_limits(px, vx, py, vy)

        wx += wx_dot * dt
        wy += wy_dot * dt
        ax_free = ax + wx * dt
        ay_free = ay + wy * dt

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
        top_pos, top_vel = self._constrain_top(top_pos, top_vel)

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
            TableAccel(x_ddot=tvx_dot, y_ddot=tvy_dot),
        )
