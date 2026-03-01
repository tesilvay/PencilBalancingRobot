import numpy as np
from simulator import Simulator
from plant import BalancerPlant
from controller import StateFeedbackController
from sim_types import (
    SystemState,
    TableCommand,
    PhysicalParams,
    SimulationResult
)

def run_simulation(
    params: PhysicalParams,
    initial_state: SystemState,
    total_time: float,
    dt: float,
    K: np.ndarray
) -> SimulationResult:

    plant = BalancerPlant(params)
    controller = StateFeedbackController(K)

    sim = Simulator(
        plant=plant,
        controller=controller,
        dt=dt
    )

    steps = int(total_time / dt)

    state_history = np.zeros((steps + 1, 8))
    acc_history = np.zeros((steps, 2))

    state = initial_state
    command = TableCommand(0.0, 0.0)

    state_history[0, :] = state.as_vector()

    for i in range(steps):
        state, command, table_acc = sim.step(state, command)

        state_history[i + 1, :] = state.as_vector()
        acc_history[i, :] = table_acc.as_vector()

        if abs(state.alpha_x) > 0.5 or abs(state.alpha_y) > 0.5:
            state_history = state_history[:i+2]
            acc_history = acc_history[:i+1]
            break

    return SimulationResult(
        state_history=state_history,
        acc_history=acc_history
    )