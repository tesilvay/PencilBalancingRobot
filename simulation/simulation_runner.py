import numpy as np
import time
from simulation.simulator import Simulator
from core.sim_types import (
    SystemState,
    TableCommand,
    PhysicalParams,
    SimulationResult
)
from system_builder import dvs_cams_connected


def initialize_histories(steps, initial_state, x_ref, y_ref):
    state_history = np.zeros((steps + 1, 8))
    state_history[0, :] = initial_state.as_vector()
    acc_history = np.zeros((steps, 2))
    cmd_history = np.zeros((steps + 1, 2))
    cmd_history[0, :] = [x_ref, y_ref]
    return state_history, acc_history, cmd_history


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
    mech=None,
    realtime: bool = False
) -> SimulationResult:

    real_mode = dvs_cams_connected(params)
    real_cams = real_mode  # run indefinitely when real DVS cams connected
    sim = Simulator(
        plant=plant,
        controller=controller,
        vision=vision,
        estimator=estimator,
        dt=dt,
        real_mode=real_mode,
    )

    run_indefinitely = real_cams and realtime
    steps = int(total_time / dt) if not run_indefinitely else 0
    paused = False  # only used when run_indefinitely (real DVS + realtime)
    if run_indefinitely:
        print("Real DVS mode: running indefinitely. Press 'q' to quit, Space to pause/unpause.")
    state = initial_state

    x_ref = params.workspace.x_ref
    y_ref = params.workspace.y_ref
    if run_indefinitely:
        state_history_list = [initial_state.as_vector()]
        acc_history_list = []
        cmd_history_list = [[x_ref, y_ref]]
        mech_history = None
    else:
        state_history, acc_history, cmd_history = initialize_histories(
            steps=steps,
            initial_state=initial_state,
            x_ref=x_ref,
            y_ref=y_ref,
        )
        if mech is not None:
            mech_history = np.full((steps + 1, 3, 2), np.nan, dtype=np.float64)
            try:
                _, _, A_g, C_g, P_g = mech.solve(np.array([initial_state.x, initial_state.y]) * 1000.0)
                mech_history[0, 0, :] = A_g
                mech_history[0, 1, :] = C_g
                mech_history[0, 2, :] = P_g
            except ValueError:
                pass
        else:
            mech_history = None

    command = TableCommand(x_ref, y_ref)

    actuator_dt, render_dt = calculate_rates(
        actuator_rate=params.hardware.servo_frequency,
        render_rate=30
    )

    # ---- Scheduler clocks (only used if realtime) ----
    if realtime:
        start_time = time.perf_counter()
        next_sim = start_time
        next_actuator = start_time
        next_render = start_time

    i = 0
    step_iter = range(steps) if not run_indefinitely else iter(int, 1)  # infinite iterator

    for _ in step_iter:

        # ---- Simulation step ----
        state, command, table_acc, measurement, pose = sim.step(state, command, realtime, actuator_dt)

        # ---- Optional real-time scheduling ----
        if realtime:

            now = time.perf_counter()
            quit_requested = False
            toggle_pause = False

            # Renderer (may return quit + toggle_pause when run_indefinitely)
            if visualizer is not None and now >= next_render:
                surfaces = None
                if vision is not None and hasattr(vision, "get_surfaces"):
                    surfaces = vision.get_surfaces()
                result = visualizer.render(
                    measurement, command=command, surfaces=surfaces,
                    paused=paused if run_indefinitely else None,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    quit_requested, toggle_pause = result
                else:
                    quit_requested = result
                next_render += render_dt

            if toggle_pause and run_indefinitely:
                paused = not paused

            # Actuator: when paused send center command, else clamped controller output
            if actuator is not None and now >= next_actuator:
                if paused and run_indefinitely:
                    center_cmd = TableCommand(x_ref, y_ref)
                    actuator.send(center_cmd)
                else:
                    command_limited = plant.clamp_command(command)
                    actuator.send(command_limited)
                next_actuator += actuator_dt

            if quit_requested and run_indefinitely:
                break

        # ---- Logging ----
        if run_indefinitely:
            state_history_list.append(state.as_vector())
            acc_history_list.append(table_acc.as_vector())
            cmd_history_list.append([command.x_des, command.y_des])
        else:
            state_history[i + 1, :] = state.as_vector()
            acc_history[i, :] = table_acc.as_vector()
            cmd_history[i + 1, :] = [command.x_des, command.y_des]
            if mech_history is not None:
                try:
                    _, _, A_g, C_g, P_g = mech.solve(np.array([state.x, state.y]) * 1000.0)
                    mech_history[i + 1, 0, :] = A_g
                    mech_history[i + 1, 1, :] = C_g
                    mech_history[i + 1, 2, :] = P_g
                except ValueError:
                    pass

        # ---- Failure condition (skip in indefinite real-time mode) ----
        if not run_indefinitely and pencil_fell(state):
            state_history = state_history[:i+2]
            acc_history = acc_history[:i+1]
            cmd_history = cmd_history[:i+2]
            if mech_history is not None:
                mech_history = mech_history[:i+2]
            break

        # ---- Real-time pacing ----
        if realtime:
            sleep_time = next_sim - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

            next_sim += dt

        i += 1
        
    if realtime:
        wall_elapsed = time.perf_counter() - start_time
        sim_steps = i + 1 if run_indefinitely else min(i + 1, steps)
        print(f"Wall time: {wall_elapsed:.3f}s")
        print(f"Simulated time: {sim_steps*dt:.3f}s")

    if run_indefinitely:
        state_history = np.array(state_history_list)
        acc_history = np.array(acc_history_list) if acc_history_list else np.zeros((0, 2))
        cmd_history = np.array(cmd_history_list) if cmd_history_list else np.zeros((0, 2))

    return SimulationResult(
        state_history=state_history,
        acc_history=acc_history,
        mech_history=mech_history,
        cmd_history=cmd_history,
    )