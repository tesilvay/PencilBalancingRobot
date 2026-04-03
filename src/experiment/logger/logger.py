from dataclasses import dataclass

import numpy as np

from src.shared import NullParams


@dataclass
class TerminalInfo:
    stabilized: bool
    settling_time: float | None


@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray
    mech_history: np.ndarray | None = None
    state_est_err_history: np.ndarray | None = None
    cmd_history: np.ndarray | None = None
    terminal: TerminalInfo | None = None


class Logger:
    def __init__(self, params: NullParams):
        self._states = None
        self._commands = None
        self._acc = None
        self._state_est_err = None

    def reset(self, initial_state, initial_command):
        # store as python lists (works for both finite + infinite)
        self._states = [initial_state.as_vector()]
        self._commands = [[initial_command.x_des, initial_command.y_des]]
        self._acc = []
        self._state_est_err = []

    def record(self, state, command, acc, state_est_err):
        self._states.append(state.as_vector())
        self._commands.append([command.x_des, command.y_des])
        self._acc.append(acc.as_vector())
        self._state_est_err.append(state_est_err)

    def get_result(self) -> SimulationResult:
        # Convert once at the end
        state_history = np.array(self._states)
        state_est_err_history = np.array(self._state_est_err)
        cmd_history = np.array(self._commands)

        if self._acc:
            acc_history = np.array(self._acc)
        else:
            acc_history = np.zeros((0, 2))

        return SimulationResult(
            state_history=state_history,
            acc_history=acc_history,
            cmd_history=cmd_history,
            state_est_err_history=state_est_err_history,
        )
