from core.sim_types import (
    PhysicalParams,
    SystemState,
    TableAccel,
    TableCommand,
    WorkspaceParams,
    clamp_table_command_to_workspace,
)
import numpy as np

# We could use the linear state-space model (A,B) for the plant dynamics
# The issue is that our plant is not actually linear
# We need to account for acc saturation, workspace limits, etc
# The controller should use A, B regardless
# But the plant should model the real physics with the nonlinear constraints

class BalancerPlant:

    def __init__(self, param: PhysicalParams):
        p = param.plant
        w = param.workspace
        self.g = p.g
        self.l = p.com_length
        self.tau = p.tau
        self.zeta = p.zeta

        self.max_acc = p.max_acc

        self.x_ref = w.x_ref
        self.y_ref = w.y_ref
        self.safe_radius = w.safe_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, state_x: SystemState, command_u: TableCommand, dt):

        # ---- unpack state ----
        x = state_x.x
        x_dot = state_x.x_dot
        alpha_x = state_x.alpha_x
        alpha_x_dot = state_x.alpha_x_dot

        y = state_x.y
        y_dot = state_x.y_dot
        alpha_y = state_x.alpha_y
        alpha_y_dot = state_x.alpha_y_dot

        # ---- clamp command (servo limits) ----
        command_u_limited = self.clamp_command(command_u)
        x_des, y_des = command_u_limited.x_des, command_u_limited.y_des

        # ---- table dynamics ----
        x_ddot = (1 / self.tau**2) * (x_des - x) \
                 - (2 * self.zeta / self.tau) * x_dot

        y_ddot = (1 / self.tau**2) * (y_des - y) \
                 - (2 * self.zeta / self.tau) * y_dot

        # ---- actuator acceleration limit ----
        x_ddot, y_ddot = self._clamp_acceleration(x_ddot, y_ddot)

        # ---- pencil dynamics ----
        alpha_x_ddot = (self.g / self.l) * alpha_x \
                       - (1 / self.l) * x_ddot

        alpha_y_ddot = (self.g / self.l) * alpha_y \
                       - (1 / self.l) * y_ddot

        # ---- integrate table ----
        x_dot += x_ddot * dt
        x += x_dot * dt

        y_dot += y_ddot * dt
        y += y_dot * dt

        # ---- enforce workspace ----
        x, x_dot, y, y_dot = self._apply_workspace_limits(
            x, x_dot, y, y_dot
        )

        # ---- integrate angles ----
        alpha_x_dot += alpha_x_ddot * dt
        alpha_x += alpha_x_dot * dt

        alpha_y_dot += alpha_y_ddot * dt
        alpha_y += alpha_y_dot * dt

        return (
            SystemState(
                x=x,
                x_dot=x_dot,
                alpha_x=alpha_x,
                alpha_x_dot=alpha_x_dot,
                y=y,
                y_dot=y_dot,
                alpha_y=alpha_y,
                alpha_y_dot=alpha_y_dot,
            ),
            TableAccel(
                x_ddot=x_ddot,
                y_ddot=y_ddot
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clamp_command(self, command_u):
        return clamp_table_command_to_workspace(
            command_u,
            WorkspaceParams(self.x_ref, self.y_ref, self.safe_radius),
        )

    def _clamp_acceleration(self, x_ddot, y_ddot):
        """
        Limit total acceleration magnitude if max_acc is set.
        If max_acc is None, do nothing.
        """
        if self.max_acc is None:
            return x_ddot, y_ddot

        acc_vec = np.array([x_ddot, y_ddot])
        norm = np.linalg.norm(acc_vec)

        if norm > self.max_acc and norm > 0:
            acc_vec = acc_vec * (self.max_acc / norm)

        return acc_vec[0], acc_vec[1]

    def _apply_workspace_limits(self, x, x_dot, y, y_dot):
        """
        Enforce circular workspace limits.

        If the table state exits the safe radius, it is projected back
        onto the boundary and outward velocity is removed.
        """

        dx = x - self.x_ref
        dy = y - self.y_ref

        dist = np.sqrt(dx*dx + dy*dy)

        if self.safe_radius is None or dist <= self.safe_radius:
            return x, x_dot, y, y_dot

        # ---- project position to boundary ----
        scale = self.safe_radius / dist
        dx *= scale
        dy *= scale

        x = self.x_ref + dx
        y = self.y_ref + dy

        # ---- remove outward velocity component ----
        normal = np.array([dx, dy]) / self.safe_radius
        vel = np.array([x_dot, y_dot])

        v_out = np.dot(vel, normal)

        if v_out > 0:
            vel = vel - v_out * normal

        return x, vel[0], y, vel[1]