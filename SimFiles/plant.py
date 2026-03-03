from sim_types import SystemState, TableCommand, TableAccel, PhysicalParams
import numpy as np


class BalancerPlant:

    def __init__(self, param: PhysicalParams):
        self.g = param.g
        self.l = param.com_length
        self.tau = param.tau
        self.zeta = param.zeta

        self.max_acc = param.max_acc

        self.x_min = param.x_min
        self.x_max = param.x_max
        self.y_min = param.y_min
        self.y_max = param.y_max

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
        x_des, y_des = self._clamp_command(command_u.x_des,
                                           command_u.y_des)

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

    def _clamp_command(self, x_des, y_des):
        """
        Clamp desired position if workspace limits exist.
        If limits are None, leave unchanged.
        """
        if self.x_min is not None:
            x_des = max(self.x_min, x_des)
        if self.x_max is not None:
            x_des = min(self.x_max, x_des)

        if self.y_min is not None:
            y_des = max(self.y_min, y_des)
        if self.y_max is not None:
            y_des = min(self.y_max, y_des)

        return x_des, y_des

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
        Enforce hard position limits.
        If a limit is None, that direction is unbounded.
        Outward velocity is zeroed at boundary.
        """

        # X limits
        if self.x_min is not None and x < self.x_min:
            x = self.x_min
            if x_dot < 0:
                x_dot = 0.0

        if self.x_max is not None and x > self.x_max:
            x = self.x_max
            if x_dot > 0:
                x_dot = 0.0

        # Y limits
        if self.y_min is not None and y < self.y_min:
            y = self.y_min
            if y_dot < 0:
                y_dot = 0.0

        if self.y_max is not None and y > self.y_max:
            y = self.y_max
            if y_dot > 0:
                y_dot = 0.0

        return x, x_dot, y, y_dot