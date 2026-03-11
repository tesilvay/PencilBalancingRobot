import numpy as np
import time
from simulation.simulator import Simulator
from core.sim_types import (
    SystemState,
    TableCommand,
    PhysicalParams,
    SimulationResult
)


def initialize_histories(steps, initial_state):

    state_history = np.zeros((steps + 1, 8))
    state_history[0, :] = initial_state.as_vector()

    acc_history = np.zeros((steps, 2))

    return state_history, acc_history


def pencil_fell(state):
    return abs(state.alpha_x) > 2.5 or abs(state.alpha_y) > 2.5


def calculate_rates(actuator_rate, render_rate):
    actuator_dt = 1 / actuator_rate
    render_dt = 1 / render_rate
    return actuator_dt, render_dt


def run_simulation(
    params: PhysicalParams,
    initial_state: SystemState,
    total_time: float,
    dt: float,
    plant,
    controller=None,
    vision=None,
    estimator=None,
    actuator=None,
    visualizer=None,
    realtime: bool = False
) -> SimulationResult:

    sim = Simulator(
        plant=plant,
        controller=controller,
        vision=vision,
        estimator=estimator,
        dt=dt
    )

    steps = int(total_time / dt)
    state = initial_state

    state_history, acc_history = initialize_histories(
        steps=steps,
        initial_state=initial_state
    )

    command = TableCommand(params.x_ref, params.y_ref)

    actuator_dt, render_dt = calculate_rates(
        actuator_rate=250,
        render_rate=30
    )

    # ---- Scheduler clocks (only used if realtime) ----
    if realtime:
        start_time = time.perf_counter()
        next_sim = start_time
        next_actuator = start_time
        next_render = start_time

    for i in range(steps):

        # ---- Simulation step ----
        state, command, table_acc, measurement, pose = sim.step(state, command, realtime, actuator_dt)

        # ---- Optional real-time scheduling ----
        if realtime:

            now = time.perf_counter()

            # Actuator
            if actuator is not None and now >= next_actuator:
                command_limited = plant.clamp_command(command)
                actuator.send(command_limited)
                next_actuator += actuator_dt

            # Renderer
            if visualizer is not None and now >= next_render:
                visualizer.render(measurement)
                next_render += render_dt

        # ---- Logging ----
        state_history[i + 1, :] = state.as_vector()
        acc_history[i, :] = table_acc.as_vector()

        # ---- Failure condition ----
        if pencil_fell(state):
            state_history = state_history[:i+2]
            acc_history = acc_history[:i+1]
            break

        # ---- Real-time pacing ----
        if realtime:
            sleep_time = next_sim - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

            next_sim += dt
        
    if realtime:
        wall_elapsed = time.perf_counter() - start_time
        print(f"Wall time: {wall_elapsed:.3f}s")
        print(f"Simulated time: {steps*dt:.3f}s")

    return SimulationResult(
        state_history=state_history,
        acc_history=acc_history
    )