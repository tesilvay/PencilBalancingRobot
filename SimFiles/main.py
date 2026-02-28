import numpy as np
import control as ct

from visualization import Visualizer3D
from simulator import Simulator
from plant import BalancerPlant
from controller import StateFeedbackController
from model import BuildLinearModel
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


def main():

    params = PhysicalParams(
        g=9.81,
        com_length=0.1,
        tau=0.04,
        zeta=0.7,
        num_states=8
    )

    initial_state = SystemState(
        x=0.0,
        x_dot=0.0,
        alpha_x=0.2,
        alpha_x_dot=0.0,
        y=0.0,
        y_dot=0.0,
        alpha_y=0.2,
        alpha_y_dot=0.0
    )

    A, B = BuildLinearModel(params)

    C_rank = np.linalg.matrix_rank(ct.ctrb(A, B))
    print(f"System controllable? {C_rank == params.num_states}")

    desired_poles = [
        -8, -10, -12, -14,
        -8, -10, -12, -14
    ]

    K = ct.place(A, B, desired_poles)

    result = run_simulation(
        params=params,
        initial_state=initial_state,
        total_time=2.0,
        dt=0.001,
        K=K
    )

    max_acc = np.max(np.abs(result.acc_history))
    print(f"Max table acceleration: {max_acc}")

    viz = Visualizer3D(result.state_history, dt=0.001)
    viz.render_video(video_speed=1, save_video=False)

    print("Initial state:", result.state_history[0])
    print("Final state:", result.state_history[-1])


if __name__ == "__main__":
    main()