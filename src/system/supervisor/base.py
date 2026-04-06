from __future__ import annotations

from dataclasses import dataclass

from numpy import deg2rad
from numpy.linalg import norm

from src.shared import ControlInput, WorkspaceParams, clamp_control_input_to_workspace


class Supervisor:
    """Base class for supervisors."""

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        """Return (controller_index, estimator_index) into System's ordered lists."""
        raise NotImplementedError

    @property
    def active_indices(self) -> tuple[int, int]:
        """Return the currently desired (controller_index, estimator_index)."""
        raise NotImplementedError

    @property
    def is_offset_latched(self) -> bool:
        """Whether the measurement offset should remain frozen."""
        return False

    @property
    def is_ready_to_run(self) -> bool:
        """Whether the supervisor has left any pre-start state."""
        return True

    @property
    def is_prestart_state(self) -> bool:
        return not self.is_ready_to_run

    @property
    def command_override(self) -> ControlInput | None:
        """Optional manual command that overrides controller output for this step."""
        return None

    @property
    def state_name(self) -> str:
        return self.__class__.__name__

    @property
    def last_transition(self) -> dict | None:
        """Metadata from last update, if available."""
        return None

    def attach_runtime(self, actuator=None, workspace=None):
        del actuator, workspace

    def handle_key(self, key: int | None) -> bool:
        del key
        return False

    def reset(self):
        pass


@dataclass(kw_only=True)
class RealStartupParams:
    centering_controller_index: int
    run_controller_index: int
    stable_threshold_deg: float
    stable_hold_s: float
    manual_step_m: float
    workspace: WorkspaceParams


class RealServoSupervisorBase(Supervisor):
    _UP_KEYS = {82, 2490368, 65362, ord("w"), ord("W")}
    _DOWN_KEYS = {84, 2621440, 65364, ord("s"), ord("S")}
    _LEFT_KEYS = {81, 2424832, 65361, ord("a"), ord("A")}
    _RIGHT_KEYS = {83, 2555904, 65363, ord("d"), ord("D")}
    _ACCEPT_KEYS = {10, 13}
    _RESET_KEYS = {ord("r"), ord("R")}

    def __init__(self, params: RealStartupParams):
        self.params = params
        self.workspace = params.workspace
        self.actuator = None
        self.state = "SERVO_CENTERING"
        self._manual_target = self._workspace_center_command()
        self._t_stable = 0.0
        self._last_transition: dict | None = None

    @property
    def is_offset_latched(self) -> bool:
        return self._measurement_offset_latched()

    @property
    def is_ready_to_run(self) -> bool:
        return self._measurement_offset_latched()

    @property
    def command_override(self) -> ControlInput | None:
        if self.state == "SERVO_CENTERING":
            return self._manual_target
        if self.state == "ACQUISITION":
            return self._workspace_center_command()
        return None

    @property
    def state_name(self) -> str:
        return self.state

    @property
    def last_transition(self) -> dict | None:
        return self._last_transition

    @property
    def manual_target(self) -> ControlInput:
        return self._manual_target

    def attach_runtime(self, actuator=None, workspace=None):
        if actuator is not None:
            self.actuator = actuator
        if workspace is not None:
            self.workspace = workspace
        self._manual_target = self._workspace_center_command()

    def handle_key(self, key: int | None) -> bool:
        if key is None or self.state != "SERVO_CENTERING":
            return False

        key_low = key & 0xFF
        if key in self._UP_KEYS or key_low in self._UP_KEYS:
            self._nudge(0.0, self.params.manual_step_m)
            return True
        if key in self._DOWN_KEYS or key_low in self._DOWN_KEYS:
            self._nudge(0.0, -self.params.manual_step_m)
            return True
        if key in self._LEFT_KEYS or key_low in self._LEFT_KEYS:
            self._nudge(-self.params.manual_step_m, 0.0)
            return True
        if key in self._RIGHT_KEYS or key_low in self._RIGHT_KEYS:
            self._nudge(self.params.manual_step_m, 0.0)
            return True
        if key in self._RESET_KEYS or key_low in self._RESET_KEYS:
            self._reset_manual_target()
            return True
        if key in self._ACCEPT_KEYS or key_low in self._ACCEPT_KEYS:
            self._accept_manual_target()
            return True
        return False

    def reset(self):
        self.state = "SERVO_CENTERING"
        self._t_stable = 0.0
        self._last_transition = None
        self._reset_manual_target()

    def _update_startup(self, x_est, dt: float) -> None:
        if self.state != "ACQUISITION":
            return
        if self._is_upright(x_est):
            self._t_stable += dt
        else:
            self._t_stable = 0.0
        if self._t_stable >= self.params.stable_hold_s:
            self._t_stable = 0.0
            self._on_upright_ready()

    def _finish_update(self, prev_state: str) -> None:
        self._last_transition = {
            "prev_state": prev_state,
            "new_state": self.state,
            "left_acquisition": (prev_state == "ACQUISITION" and self.state != "ACQUISITION"),
        }

    def _workspace_center_command(self) -> ControlInput:
        return ControlInput(
            px_cmd=float(self.workspace.x_ref),
            py_cmd=float(self.workspace.y_ref),
        )

    def _reset_manual_target(self) -> None:
        self._manual_target = self._workspace_center_command()
        self._apply_workspace_offset(0.0, 0.0)

    def _nudge(self, dx: float, dy: float) -> None:
        candidate = ControlInput(
            px_cmd=self._manual_target.px_cmd + dx,
            py_cmd=self._manual_target.py_cmd + dy,
        )
        self._manual_target = clamp_control_input_to_workspace(candidate, self.workspace)

    def _accept_manual_target(self) -> None:
        dx = float(self._manual_target.px_cmd - self.workspace.x_ref)
        dy = float(self._manual_target.py_cmd - self.workspace.y_ref)
        self._apply_workspace_offset(dx, dy)
        self.state = "ACQUISITION"
        self._t_stable = 0.0

    def _apply_workspace_offset(self, dx: float, dy: float) -> None:
        actuator = self.actuator
        if actuator is not None and hasattr(actuator, "set_workspace_offset"):
            actuator.set_workspace_offset(dx, dy)

    def _is_upright(self, x_est) -> bool:
        return norm([float(x_est.ax), float(x_est.ay)]) < deg2rad(self.params.stable_threshold_deg)

    def _measurement_offset_latched(self) -> bool:
        raise NotImplementedError

    def _on_upright_ready(self) -> None:
        raise NotImplementedError
