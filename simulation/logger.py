import numpy as np
from core.sim_types import SimulationResult


class Logger:
    def __init__(self):
        self._states = None
        self._commands = None
        self._acc = None

    def reset(self, initial_state, initial_command):
        # store as python lists (works for both finite + infinite)
        self._states = [initial_state.as_vector()]
        self._commands = [[initial_command.x_des, initial_command.y_des]]
        self._acc = []

    def record(self, state, command, acc):
        self._states.append(state.as_vector())
        self._commands.append([command.x_des, command.y_des])
        self._acc.append(acc.as_vector())

    def get_result(self) -> SimulationResult:
        # Convert once at the end
        state_history = np.array(self._states)
        cmd_history = np.array(self._commands)

        if self._acc:
            acc_history = np.array(self._acc)
        else:
            acc_history = np.zeros((0, 2))

        return SimulationResult(
            state_history=state_history,
            acc_history=acc_history,
            cmd_history=cmd_history,
        )
