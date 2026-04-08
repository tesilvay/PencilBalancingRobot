from dataclasses import dataclass, field

import numpy as np

from .base import BasePlant
from src.shared import (
    ControlInput,
    PlantParams,
    State,
    TableAccel,
    WorkspaceParams,
    default_plant,
    default_workspace,
)


@dataclass
class AccelFollowParams:
    plant: PlantParams = field(default_factory=default_plant)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    base_mode: str = "ideal"


ACCEL_FOLLOW_PRESETS = {
    "default": {
        "base_mode": "ideal",
    },
    "lagged": {
        "base_mode": "lagged",
    },
}


class AccelFollowPlant(BasePlant):
    """
    Simulation plant for the acceleration-command architecture.

    Publicly it still accepts position commands, but interprets them as the
    controller's internally integrated table trajectory. In ideal mode the
    plant recovers table velocity and acceleration directly from command
    differences. In lagged mode it lets a second-order table model track the
    commanded position for comparison.
    """

    def __init__(self, params: AccelFollowParams):
        p = params.plant
        w = params.workspace

        super().__init__(w, p.max_acc)

        self.g = p.g
        self.l = p.com_length
        self.tau = p.tau
        self.zeta = p.zeta
        self.base_mode = params.base_mode

        self.reset()

    def reset(self):
        self._last_cmd_pos = np.zeros(2, dtype=float)
        self._last_cmd_vel = np.zeros(2, dtype=float)
        self._initialized = False

    def _bootstrap_history(self, state_x: State, u_cmd: ControlInput):
        del u_cmd
        self._last_cmd_pos = np.array([state_x.px, state_x.py], dtype=float)
        self._last_cmd_vel = np.array([state_x.vx, state_x.vy], dtype=float)
        self._initialized = True

    def _ideal_table_step(self, state_x: State, u_cmd: ControlInput, dt: float):
        cmd_pos = np.array([u_cmd.px_cmd, u_cmd.py_cmd], dtype=float)
        cmd_vel = (cmd_pos - self._last_cmd_pos) / dt
        cmd_acc = (cmd_vel - self._last_cmd_vel) / dt
        cmd_acc[0], cmd_acc[1] = self.clamp_acceleration(cmd_acc[0], cmd_acc[1])

        px = float(cmd_pos[0])
        py = float(cmd_pos[1])
        vx = float(cmd_vel[0])
        vy = float(cmd_vel[1])
        vx_dot = float(cmd_acc[0])
        vy_dot = float(cmd_acc[1])

        self._last_cmd_pos = cmd_pos
        self._last_cmd_vel = cmd_vel

        return px, vx, vx_dot, py, vy, vy_dot

    def _lagged_table_step(self, state_x: State, u_cmd: ControlInput, dt: float):
        px = state_x.px
        vx = state_x.vx
        py = state_x.py
        vy = state_x.vy

        px_cmd = float(u_cmd.px_cmd)
        py_cmd = float(u_cmd.py_cmd)

        vx_dot = (1.0 / self.tau**2) * (px_cmd - px) - (2.0 * self.zeta / self.tau) * vx
        vy_dot = (1.0 / self.tau**2) * (py_cmd - py) - (2.0 * self.zeta / self.tau) * vy
        vx_dot, vy_dot = self.clamp_acceleration(vx_dot, vy_dot)

        vx += vx_dot * dt
        px += vx * dt
        vy += vy_dot * dt
        py += vy * dt

        self._last_cmd_pos = np.array([px_cmd, py_cmd], dtype=float)
        self._last_cmd_vel = np.array([vx, vy], dtype=float)

        return px, vx, vx_dot, py, vy, vy_dot

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        if not self._initialized:
            self._bootstrap_history(state_x, u_cmd)
            return (
                state_x,
                TableAccel(x_ddot=0.0, y_ddot=0.0),
            )

        if self.base_mode == "lagged":
            px, vx, vx_dot, py, vy, vy_dot = self._lagged_table_step(state_x, u_cmd, dt)
        else:
            px, vx, vx_dot, py, vy, vy_dot = self._ideal_table_step(state_x, u_cmd, dt)

        px, vx, py, vy = self.apply_workspace_limits(px, vx, py, vy)

        ax = state_x.ax
        wx = state_x.wx
        ay = state_x.ay
        wy = state_x.wy

        wx_dot = (self.g / self.l) * ax - (1.0 / self.l) * vx_dot
        wy_dot = (self.g / self.l) * ay - (1.0 / self.l) * vy_dot

        wx += wx_dot * dt
        ax += wx * dt
        wy += wy_dot * dt
        ay += wy * dt

        ax = float(np.clip(ax, -np.pi / 2, np.pi / 2))
        ay = float(np.clip(ay, -np.pi / 2, np.pi / 2))

        return (
            State(
                px=float(px), vx=float(vx),
                ax=ax, wx=float(wx),
                py=float(py), vy=float(vy),
                ay=ay, wy=float(wy),
            ),
            TableAccel(x_ddot=float(vx_dot), y_ddot=float(vy_dot)),
        )
