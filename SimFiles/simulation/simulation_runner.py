import numpy as np
import time
from simulation.simulator import Simulator
from core.plant import BalancerPlant
from core.controller import NullController
from core.sim_types import (
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
    controller=None,
    vision=None,
    estimator=None,
    actuator=None
) -> SimulationResult:

    plant = BalancerPlant(params)
    if controller is None:
        controller = NullController()

    sim = Simulator(
        plant=plant,
        controller=controller,
        vision=vision,
        estimator=estimator,
        dt=dt
    )

    steps = int(total_time / dt)

    state_history = np.zeros((steps + 1, 8))
    acc_history = np.zeros((steps, 2))

    state = initial_state
    command = TableCommand(0.0, 0.0)

    state_history[0, :] = state.as_vector()

    # real-time scheduler
    if actuator is not None:
        next_time = time.perf_counter()
        
    for i in range(steps):
        state, command, table_acc = sim.step(state, command)
        
        if actuator is not None:
            command_limited = plant.clamp_command(command)
            actuator.send(command_limited)
            
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # optional: track missed deadlines
                pass
            
            next_time += dt

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