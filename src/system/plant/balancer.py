from dataclasses import dataclass, field

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
class BalancerParams:
    plant:      PlantParams     = field(default_factory=default_plant)
    workspace:  WorkspaceParams = field(default_factory=default_workspace)


BALANCER_PRESETS = {
    "default": {},
}


class BalancerPlant:

    def __init__(self, params: BalancerParams):
        p = params.plant
        w = params.workspace

        self.g    = p.g
        self.l    = p.com_length
        self.tau  = p.tau
        self.zeta = p.zeta
        self.max_acc = p.max_acc

        self.x_ref       = w.x_ref
        self.y_ref       = w.y_ref
        self.safe_radius = w.safe_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # find acceleration and clamp it to realistic values
        vx_dot = (1 / self.tau**2) * (px_cmd - px) - (2 * self.zeta / self.tau) * vx
        vy_dot = (1 / self.tau**2) * (py_cmd - py) - (2 * self.zeta / self.tau) * vy
        vx_dot, vy_dot = self._clamp_acceleration(vx_dot, vy_dot)
        
        # find angular acceleration
        wx_dot = (self.g / self.l) * ax - (1 / self.l) * vx_dot
        wy_dot = (self.g / self.l) * ay - (1 / self.l) * vy_dot

        # based on acc, get vel and pos too
        vx += vx_dot * dt
        px += vx  * dt

        vy += vy_dot * dt
        py += vy  * dt

        # make sure we stay inside the workspace
        px, vx, py, vy = self._apply_workspace_limits(px, vx, py, vy)

        # based on angular acc, get w and ang too
        wx += wx_dot * dt
        ax  += wx  * dt

        wy += wy_dot * dt
        ay  += wy  * dt

        # make sure we limit angle to being flat on table
        ax = float(np.clip(ax, -np.pi / 2, np.pi / 2))
        ay = float(np.clip(ay, -np.pi / 2, np.pi / 2))

        return (
            State(
                px=px, vx=vx,
                ax=ax, wx=wx,
                py=py, vy=vy,
                ay=ay, wy=wy,
            ),
            TableAccel(x_ddot=vx_dot, y_ddot=vy_dot),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_acceleration(self, vx_dot, vy_dot):
        if self.max_acc is None:
            return vx_dot, vy_dot

        acc_vec = np.array([vx_dot, vy_dot])
        norm    = np.linalg.norm(acc_vec)

        if norm > self.max_acc and norm > 0:
            acc_vec = acc_vec * (self.max_acc / norm)

        return acc_vec[0], acc_vec[1]

    def _apply_workspace_limits(self, x, vx, y, vy):
        dx   = x - self.x_ref
        dy   = y - self.y_ref
        dist = np.sqrt(dx * dx + dy * dy)

        if self.safe_radius is None or dist <= self.safe_radius:
            return x, vx, y, vy

        scale = self.safe_radius / dist
        dx   *= scale
        dy   *= scale
        x     = self.x_ref + dx
        y     = self.y_ref + dy

        normal = np.array([dx, dy]) / self.safe_radius
        vel    = np.array([vx, vy])
        v_out  = np.dot(vel, normal)

        if v_out > 0:
            vel = vel - v_out * normal

        return x, vel[0], y, vel[1]
