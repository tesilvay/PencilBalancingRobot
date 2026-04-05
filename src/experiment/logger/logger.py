from dataclasses import dataclass

import numpy as np

from src.shared import NullParams, StepData, TableAccel


@dataclass
class TerminalInfo:
    stabilized: bool
    settling_time: float | None


@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray
    mech_history: np.ndarray | None = None
    innovation_history: np.ndarray | None = None
    cmd_history: np.ndarray | None = None
    terminal: TerminalInfo | None = None


def _acc_to_row(acc: TableAccel | np.ndarray) -> np.ndarray:
    if isinstance(acc, TableAccel):
        return acc.as_vector()
    a = np.asarray(acc, dtype=float).reshape(-1)
    if a.size != 2:
        raise ValueError(f"acc must have 2 elements, got shape {a.shape}")
    return a


def _innovation_to_row(innovation: np.ndarray | None) -> np.ndarray:
    if innovation is None:
        return np.full(4, np.nan, dtype=float)
    a = np.asarray(innovation, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(f"innovation must have 4 elements, got shape {a.shape}")
    return a


class Logger:
    def __init__(self, params: NullParams):
        self._states = None
        self._commands = None
        self._acc = None
        self._innovation = None

    def reset(self, initial_state, initial_command):
        # store as python lists (works for both finite + infinite)
        self._states = [initial_state.as_vector()]
        self._commands = [[initial_command.px_cmd, initial_command.py_cmd]]
        self._acc = []
        self._innovation = []

    def record(self, step_data: StepData):
        self._states.append(step_data.x.as_vector())
        self._commands.append([step_data.u.px_cmd, step_data.u.py_cmd])
        self._acc.append(_acc_to_row(step_data.acc))
        self._innovation.append(_innovation_to_row(step_data.innovation))

    def get_result(self) -> SimulationResult:
        # Convert once at the end
        state_history = np.array(self._states)
        cmd_history = np.array(self._commands)

        if self._acc:
            acc_history = np.array(self._acc)
        else:
            acc_history = np.zeros((0, 2))

        if self._innovation:
            innovation_history = np.array(self._innovation)
        else:
            innovation_history = np.zeros((0, 4))

        return SimulationResult(
            state_history=state_history,
            acc_history=acc_history,
            cmd_history=cmd_history,
            innovation_history=innovation_history,
        )
