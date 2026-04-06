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
class PlacingParams:
    """
    Human is holding the pencil on the table: no friction coupling from table
    motion into the pencil. Table acceleration is still computed from the command
    (second-order tracking). Pencil position and tilt evolve from a hand/tremor model.
    """

    plant: PlantParams = field(default_factory=default_plant)
    workspace: WorkspaceParams = field(default_factory=default_workspace)

    # Second-order hand model on pencil tip (px, py) about workspace reference.
    tip_omega: float = 12.0
    tip_zeta: float = 0.85
    tip_accel_noise_std: float = 0.15  # m/s^2, driving noise on vx_dot, vy_dot

    # Second-order tremor on tilt (ax, ay) about upright (0, 0).
    angle_omega: float = 10.0
    angle_zeta: float = 0.9
    angle_accel_noise_std: float = 8.0  # rad/s^2

    rng_seed: int | None = None


PLACING_PRESETS = {
    "steady_hands": {
        "tip_accel_noise_std": 0.08,
        "angle_accel_noise_std": 2.0,
    },
    "shaky": {
        "tip_accel_noise_std": 0.25,
        "angle_accel_noise_std": 8.0,
    },
}


class PlacingPlant:
    """Simulate pencil pose while a human holds it; table command does not move the pencil."""

    def __init__(self, params: PlacingParams):
        p = params.plant
        w = params.workspace

        self.tau = p.tau
        self.zeta = p.zeta
        self.max_acc = p.max_acc

        self.x_ref = w.x_ref
        self.y_ref = w.y_ref
        self.safe_radius = w.safe_radius

        self.tip_omega = params.tip_omega
        self.tip_zeta = params.tip_zeta
        self.tip_accel_noise_std = params.tip_accel_noise_std

        self.angle_omega = params.angle_omega
        self.angle_zeta = params.angle_zeta
        self.angle_accel_noise_std = params.angle_accel_noise_std

        self._rng = np.random.default_rng(params.rng_seed)

        self._tx = float(w.x_ref)
        self._ty = float(w.y_ref)
        self._tvx = 0.0
        self._tvy = 0.0

    def reset(self) -> None:
        """Re-sync internal table state (e.g. new trial). Pencil state comes from System.reset."""
        self._tx = float(self.x_ref)
        self._ty = float(self.y_ref)
        self._tvx = 0.0
        self._tvy = 0.0

    def step(self, state_x: State, u_cmd: ControlInput, dt: float):
        px_cmd, py_cmd = u_cmd.px_cmd, u_cmd.py_cmd

        # --- Table (command tracking only; not coupled into pencil state) ---
        tvx_dot = (1 / self.tau**2) * (px_cmd - self._tx) - (2 * self.zeta / self.tau) * self._tvx
        tvy_dot = (1 / self.tau**2) * (py_cmd - self._ty) - (2 * self.zeta / self.tau) * self._tvy
        tvx_dot, tvy_dot = self._clamp_acceleration(tvx_dot, tvy_dot)

        self._tvx += tvx_dot * dt
        self._tx += self._tvx * dt
        self._tvy += tvy_dot * dt
        self._ty += self._tvy * dt

        self._tx, self._tvx, self._ty, self._tvy = self._apply_workspace_limits(
            self._tx, self._tvx, self._ty, self._tvy
        )

        # --- Pencil held by hand: tip and angles independent of table accel ---
        px, vx = state_x.px, state_x.vx
        py, vy = state_x.py, state_x.vy
        ax, wx = state_x.ax, state_x.wx
        ay, wy = state_x.ay, state_x.wy

        wn = self.tip_omega
        zt = self.tip_zeta
        n_scale_tip = self.tip_accel_noise_std / np.sqrt(dt) if dt > 0 else 0.0
        vx_dot = (
            -(wn**2) * (px - self.x_ref)
            - 2 * zt * wn * vx
            + float(self._rng.normal(0.0, n_scale_tip))
        )
        vy_dot = (
            -(wn**2) * (py - self.y_ref)
            - 2 * zt * wn * vy
            + float(self._rng.normal(0.0, n_scale_tip))
        )

        wa = self.angle_omega
        za = self.angle_zeta
        n_scale_ang = self.angle_accel_noise_std / np.sqrt(dt) if dt > 0 else 0.0
        wx_dot = (
            -(wa**2) * ax
            - 2 * za * wa * wx
            + float(self._rng.normal(0.0, n_scale_ang))
        )
        wy_dot = (
            -(wa**2) * ay
            - 2 * za * wa * wy
            + float(self._rng.normal(0.0, n_scale_ang))
        )

        vx += vx_dot * dt
        px += vx * dt
        vy += vy_dot * dt
        py += vy * dt

        wx += wx_dot * dt
        ax += wx * dt
        wy += wy_dot * dt
        ay += wy * dt

        px, vx, py, vy = self._apply_workspace_limits(px, vx, py, vy)

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

    def _clamp_acceleration(self, vx_dot, vy_dot):
        if self.max_acc is None:
            return vx_dot, vy_dot

        acc_vec = np.array([vx_dot, vy_dot])
        norm = np.linalg.norm(acc_vec)

        if norm > self.max_acc and norm > 0:
            acc_vec = acc_vec * (self.max_acc / norm)

        return float(acc_vec[0]), float(acc_vec[1])

    def _apply_workspace_limits(self, x, vx, y, vy):
        if self.safe_radius is None:
            return x, vx, y, vy

        dx = x - self.x_ref
        dy = y - self.y_ref
        dist = np.sqrt(dx * dx + dy * dy)

        if dist <= self.safe_radius:
            return x, vx, y, vy

        scale = self.safe_radius / dist
        dx *= scale
        dy *= scale
        x = self.x_ref + dx
        y = self.y_ref + dy

        normal = np.array([dx, dy]) / self.safe_radius
        vel = np.array([vx, vy])
        v_out = np.dot(vel, normal)

        if v_out > 0:
            vel = vel - v_out * normal

        return x, vel[0], y, vel[1]
