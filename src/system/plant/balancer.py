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

    def step(self, state_x: State, command_u: ControlInput, dt: float):

        x         = state_x.px
        x_dot     = state_x.vx
        alpha_x   = state_x.ax
        alpha_x_dot = state_x.wx

        y         = state_x.py
        y_dot     = state_x.vy
        alpha_y   = state_x.ay
        alpha_y_dot = state_x.wy

        command_u_limited = self.clamp_command(command_u)
        px_cmd, py_cmd = command_u_limited.px_cmd, command_u_limited.py_cmd

        x_ddot = (1 / self.tau**2) * (px_cmd - x) - (2 * self.zeta / self.tau) * x_dot
        y_ddot = (1 / self.tau**2) * (py_cmd - y) - (2 * self.zeta / self.tau) * y_dot

        x_ddot, y_ddot = self._clamp_acceleration(x_ddot, y_ddot)

        alpha_x_ddot = (self.g / self.l) * alpha_x - (1 / self.l) * x_ddot
        alpha_y_ddot = (self.g / self.l) * alpha_y - (1 / self.l) * y_ddot

        x_dot += x_ddot * dt
        x     += x_dot  * dt

        y_dot += y_ddot * dt
        y     += y_dot  * dt

        x, x_dot, y, y_dot = self._apply_workspace_limits(x, x_dot, y, y_dot)

        alpha_x_dot += alpha_x_ddot * dt
        alpha_x     += alpha_x_dot  * dt

        alpha_y_dot += alpha_y_ddot * dt
        alpha_y     += alpha_y_dot  * dt

        alpha_x = float(np.clip(alpha_x, -np.pi / 2, np.pi / 2))
        alpha_y = float(np.clip(alpha_y, -np.pi / 2, np.pi / 2))

        return (
            State(
                px=x, vx=x_dot,
                ax=alpha_x, wx=alpha_x_dot,
                py=y, vy=y_dot,
                ay=alpha_y, wy=alpha_y_dot,
            ),
            TableAccel(x_ddot=x_ddot, y_ddot=y_ddot),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clamp_command(self, command_u):
        px_cmd, py_cmd = command_u.px_cmd, command_u.py_cmd
        safe_radius  = self.safe_radius

        if safe_radius is None:
            return ControlInput(px_cmd, py_cmd)

        dx   = px_cmd - self.x_ref
        dy   = py_cmd - self.y_ref
        dist = float(np.sqrt(dx * dx + dy * dy))

        if dist > safe_radius and dist > 0:
            scale = safe_radius / dist
            px_cmd = self.x_ref + dx * scale
            py_cmd = self.y_ref + dy * scale

        return ControlInput(px_cmd, py_cmd)

    def _clamp_acceleration(self, x_ddot, y_ddot):
        if self.max_acc is None:
            return x_ddot, y_ddot

        acc_vec = np.array([x_ddot, y_ddot])
        norm    = np.linalg.norm(acc_vec)

        if norm > self.max_acc and norm > 0:
            acc_vec = acc_vec * (self.max_acc / norm)

        return acc_vec[0], acc_vec[1]

    def _apply_workspace_limits(self, x, x_dot, y, y_dot):
        dx   = x - self.x_ref
        dy   = y - self.y_ref
        dist = np.sqrt(dx * dx + dy * dy)

        if self.safe_radius is None or dist <= self.safe_radius:
            return x, x_dot, y, y_dot

        scale = self.safe_radius / dist
        dx   *= scale
        dy   *= scale
        x     = self.x_ref + dx
        y     = self.y_ref + dy

        normal = np.array([dx, dy]) / self.safe_radius
        vel    = np.array([x_dot, y_dot])
        v_out  = np.dot(vel, normal)

        if v_out > 0:
            vel = vel - v_out * normal

        return x, vel[0], y, vel[1]
