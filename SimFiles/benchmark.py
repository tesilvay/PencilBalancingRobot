import numpy as np
from simulation_runner import run_simulation
from sim_types import SystemState, PhysicalParams

def evaluate_stability(result, dt, tol=0.02, hold_time=0.2):

    alpha_x = result.state_history[:, 2]
    alpha_y = result.state_history[:, 6]

    inside = (np.abs(alpha_x) < tol) & (np.abs(alpha_y) < tol)

    hold_steps = int(hold_time / dt)

    for i in range(len(inside) - hold_steps):
        if np.all(inside[i:i+hold_steps]):
            return True, i * dt

    return False, None

def region_mapping(params, K, dt=0.001, total_time=2.0, n_trials=200):

    results = []

    for _ in range(n_trials):
        
        print(f"Trial #{_}")

        alpha_x = np.random.uniform(-0.5, 0.5)
        alpha_x_dot = np.random.uniform(-0.5, 0.5)

        alpha_y = np.random.uniform(-0.5, 0.5)
        alpha_y_dot = np.random.uniform(-0.5, 0.5)

        initial_state = SystemState(
            x=0.0,
            x_dot=0.0,
            alpha_x=alpha_x,
            alpha_x_dot=alpha_x_dot,
            y=0.0,
            y_dot=0.0,
            alpha_y=alpha_y,
            alpha_y_dot=alpha_y_dot
        )

        result = run_simulation(
            params=params,
            initial_state=initial_state,
            total_time=total_time,
            dt=dt,
            K=K
        )

        stabilized, settling_time = evaluate_stability(result, dt)
        max_acc = np.max(np.abs(result.acc_history))

        results.append({
            "alpha_x0": alpha_x,
            "alpha_x_dot0": alpha_x_dot,
            "stabilized": stabilized,
            "settling_time": settling_time,
            "max_acc": max_acc
        })

    return results