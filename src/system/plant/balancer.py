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
class BalancerParams:
    disturbance_cart_force_x: float
    plant:      PlantParams     = field(default_factory=default_plant)
    workspace:  WorkspaceParams = field(default_factory=default_workspace)


BALANCER_PRESETS = {
    "default": {
        "disturbance_cart_force_x": 0,
    },
    "noise": {
        "disturbance_cart_force_x": 5,
    },
}


class BalancerPlant(BasePlant):

    def __init__(self, params: BalancerParams):
        
        p = params.plant
        w = params.workspace
        self.disturbance_cart_force_x = params.disturbance_cart_force_x
        
        super().__init__(w, p.max_acc)

        self.g    = p.g
        self.l    = p.com_length
        self.tau  = p.tau
        self.zeta = p.zeta

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

        # table dynamics
        vx_dot = (1 / self.tau**2) * (px_cmd - px) - (2 * self.zeta / self.tau) * vx
        vy_dot = (1 / self.tau**2) * (py_cmd - py) - (2 * self.zeta / self.tau) * vy
        
        vx_dot += self.disturbance_cart_force_x
        
        vx_dot, vy_dot = self.clamp_acceleration(vx_dot, vy_dot)
        
        # pencil dynamics
        wx_dot = (self.g / self.l) * ax - (1 / self.l) * vx_dot
        wy_dot = (self.g / self.l) * ay - (1 / self.l) * vy_dot

        # based on acc, get vel and pos too
        vx += vx_dot * dt
        px += vx  * dt

        vy += vy_dot * dt
        py += vy  * dt

        # make sure we stay inside the workspace
        px, vx, py, vy = self.apply_workspace_limits(px, vx, py, vy)

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