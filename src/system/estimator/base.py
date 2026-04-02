import numpy as np
from core.sim_types import SystemState, PoseMeasurement, TableCommand

from perception.estimator_diagnostics import EstimatorDiagnosticSnapshot


class BaseEstimator:
    def __init__(self):
        self._last_diagnostic_snapshot: EstimatorDiagnosticSnapshot | None = None
        self._diag_step_idx = 0
        self._diag_t_s = 0.0
        self._diag_measurement_fresh = True
        self._diag_z_changed = True

    def set_diagnostics_context(
        self,
        step_idx: int,
        t_s: float,
        measurement_fresh: bool,
        z_changed: bool,
    ) -> None:
        self._diag_step_idx = step_idx
        self._diag_t_s = t_s
        self._diag_measurement_fresh = measurement_fresh
        self._diag_z_changed = z_changed

    def get_last_diagnostics(self) -> EstimatorDiagnosticSnapshot | None:
        return self._last_diagnostic_snapshot

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:
        raise NotImplementedError

    def reset(self):
        self._last_diagnostic_snapshot = None

    def _print_state(self, state):
        print(
            f"x={state.x*1000:+.2f} mm, x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(state.alpha_x):+.2f}°, ax_dot={np.rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y={state.y*1000:+.2f} mm, y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(state.alpha_y):+.2f}°, ay_dot={np.rad2deg(state.alpha_y_dot):+.2f}°/s"
        )

    def _print_vel(self, state):
        print(
            f"x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax_dot={np.rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay_dot={np.rad2deg(state.alpha_y_dot):+.2f}°/s"
        )

    def _print_pose(self, pose):
        x = pose[0, 0]
        ax = pose[1, 0]
        y = pose[2, 0]
        ay = pose[3, 0]
        print(
            f"pose:   "
            f"x={x*1000:+.2f} mm, "
            f"ax={np.rad2deg(ax):+.2f}° | "
            f"y={y*1000:+.2f} mm, "
            f"ay={np.rad2deg(ay):+.2f}°"
        )

    def _print_est(self, est):
        x = est[0, 0]
        x_dot = est[1, 0]
        ax = est[2, 0]
        ax_dot = est[3, 0]
        y = est[4, 0]
        y_dot = est[5, 0]
        ay = est[6, 0]
        ay_dot = est[7, 0]

        print(
            f"z:  "
            f"x={x*1000:+.2f} mm, x_dot={x_dot*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(ax):+.2f}°, ax_dot={np.rad2deg(ax_dot):+.2f}°/s | "
            f"y={y*1000:+.2f} mm, y_dot={y_dot*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(ay):+.2f}°, ay_dot={np.rad2deg(ay_dot):+.2f}°/s"
        )

    def _print_est_x_hat(self, est):
        x = est[0, 0]
        x_dot = est[1, 0]
        ax = est[2, 0]
        ax_dot = est[3, 0]
        y = est[4, 0]
        y_dot = est[5, 0]
        ay = est[6, 0]
        ay_dot = est[7, 0]

        print(
            f"x_hat:  "
            f"x={x*1000:+.2f} mm, x_dot={x_dot*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(ax):+.2f}°, ax_dot={np.rad2deg(ax_dot):+.2f}°/s | "
            f"y={y*1000:+.2f} mm, y_dot={y_dot*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(ay):+.2f}°, ay_dot={np.rad2deg(ay_dot):+.2f}°/s"
        )
