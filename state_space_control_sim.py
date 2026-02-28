from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import control as ct
import os
from datetime import datetime

# -----------------------------
# PHYSICAL PARAMETERS
# -----------------------------

g = 9.81           # gravity
L = 0.2            # pencil length
l = L / 2          # distance from pivot to center of mass

tau = 0.04         # actuator time constant
zeta = 0.7         # actuator damping ratio
num_states = 8     # number of states for the pencil


# -----------------------------
# 3D pencil state
# -----------------------------
    
@dataclass
class PencilState:
    x: float
    x_dot: float
    alpha_x: float
    alpha_x_dot: float
    y: float
    y_dot: float
    alpha_y: float
    alpha_y_dot: float

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.x,
            self.x_dot,
            self.alpha_x,
            self.alpha_x_dot,
            self.y,
            self.y_dot,
            self.alpha_y,
            self.alpha_y_dot
        ])


# -----------------------------
# Filters
# -----------------------------


    

# -----------------------------
# Pure P Controller (position + slope only)
# -----------------------------
    
class StateFeedbackController:
    def __init__(self, K: np.ndarray):
        """
        K must be shape (2, 8)
        First row -> x_des
        Second row -> y_des
        """
        self.K = K

    def compute(self, state: PencilState) -> tuple[float, float]:
        x_vec = state.as_vector()
        u = -self.K @ x_vec
        return u[0], u[1]
    


# -----------------------------
# Simulation Plant
# -----------------------------

class PencilPlant:

    def __init__(self):
        self.g = 9.81
        self.l = l

        # pencil states
        self.alpha_x = 0.3  # small initial tilt
        self.alpha_y = -0.2
        self.alpha_x_dot = 0.0
        self.alpha_y_dot = 0.0

        # table states
        self.x = 0.04
        self.y = 0.08
        self.x_dot = 0.0
        self.y_dot = 0.0

        # table dynamics params
        self.tau = tau
        self.zeta = zeta

    def step(self, x_des, y_des, dt):

        # ---- table dynamics ----
        x_ddot = (1/self.tau**2)*(x_des - self.x) - (2*self.zeta/self.tau)*self.x_dot
        y_ddot = (1/self.tau**2)*(y_des - self.y) - (2*self.zeta/self.tau)*self.y_dot

        self.x_dot += x_ddot * dt
        self.y_dot += y_ddot * dt
        self.x += self.x_dot * dt
        self.y += self.y_dot * dt

        # ---- pencil dynamics ----
        alpha_x_ddot = (self.g/self.l)*self.alpha_x - (1/self.l)*x_ddot
        alpha_y_ddot = (self.g/self.l)*self.alpha_y - (1/self.l)*y_ddot

        self.alpha_x_dot += alpha_x_ddot * dt
        self.alpha_y_dot += alpha_y_ddot * dt
        self.alpha_x += self.alpha_x_dot * dt
        self.alpha_y += self.alpha_y_dot * dt

        return PencilState(
            x=self.x,
            x_dot=self.x_dot,
            alpha_x=self.alpha_x,
            alpha_x_dot=self.alpha_x_dot,
            y=self.y,
            y_dot=self.y_dot,
            alpha_y=self.alpha_y,
            alpha_y_dot=self.alpha_y_dot
        )
    
        
        
# -----------------------------
# Simulation
# -----------------------------
class Simulator:

    def __init__(self, plant, controller, dt=0.001):
        self.plant = plant
        self.controller = controller
        self.dt = dt

        self.x_des = 0.0
        self.y_des = 0.0

    def step(self):

        # advance plant using last control
        state = self.plant.step(
            self.x_des,
            self.y_des,
            self.dt
        )

        # compute new control from updated state
        self.x_des, self.y_des = self.controller.compute(state)

        return state


# -----------------------------
# 3D Visualization
# -----------------------------


class Visualizer3D:

    def __init__(self, history, dt, L=0.2, fps=60):

        self.history = history
        self.dt = dt
        self.L = L
        self.fps = fps
        self.frame_period = 1.0 / fps
        self.total_sim_time = history.shape[0] * dt

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(-0.15, 0.15)
        self.ax.set_ylim(-0.15, 0.15)
        self.ax.set_zlim(0, L)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.table_plot, = self.ax.plot([], [], [], 'k-', linewidth=2)
        self.pencil_plot, = self.ax.plot([], [], [], 'r-', linewidth=3)

        self.info_text = self.ax.text2D(
            0.02, 0.95,
            "",
            transform=self.ax.transAxes
        )

        plt.ion()
        plt.show()

    # -------------------------------------------------
    # Render a single frame
    # -------------------------------------------------
    def render_frame(self, state, sim_time):

        x = state[0]
        alpha_x = state[2]
        y = state[4]
        alpha_y = state[6]

        size = 0.05

        corners_x = [x - size, x + size, x + size, x - size, x - size]
        corners_y = [y - size, y - size, y + size, y + size, y - size]
        corners_z = [0, 0, 0, 0, 0]

        self.table_plot.set_data(corners_x, corners_y)
        self.table_plot.set_3d_properties(corners_z)

        base = np.array([x, y, 0.0])
        direction = np.array([alpha_x, alpha_y, 1.0])
        direction /= np.linalg.norm(direction)
        tip = base + direction * self.L

        self.pencil_plot.set_data(
            [base[0], tip[0]],
            [base[1], tip[1]]
        )
        self.pencil_plot.set_3d_properties(
            [base[2], tip[2]]
        )

        self.info_text.set_text(
            f"Sim Time: {sim_time:.3f} s\n"
            f"x: {x:.3f}\n"
            f"y: {y:.3f}\n"
            f"αx: {alpha_x:.3f}\n"
            f"αy: {alpha_y:.3f}"
        )

        plt.draw()
        plt.pause(0.001)

    # -------------------------------------------------
    # Playback
    # -------------------------------------------------
    def render_video(self, video_speed=1.0, filename=None):
        """
        video_speed:
            1.0 → real time
            2.0 → 2x faster
            0.5 → half speed
        """

        os.makedirs("simulation_video", exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_video/pencil_sim_{timestamp}.mp4"

        writer = FFMpegWriter(fps=self.fps)

        total_sim_time = self.history.shape[0] * self.dt

        print("Rendering video...")

        with writer.saving(self.fig, filename, dpi=100):

            frame_number = 0

            while True:

                # simulated time for this frame
                sim_time = frame_number * self.frame_period * video_speed

                if sim_time >= total_sim_time:
                    break

                sim_index = int(round(sim_time / self.dt))

                if sim_index >= len(self.history):
                    break

                state = self.history[sim_index]

                # render frame
                self.render_frame(state, sim_time)

                writer.grab_frame()

                frame_number += 1

        print(f"Video saved to: {filename}")
 

def BuildLinearModel():
    # -----------------------------
    # STATE VECTOR DEFINITION (ONE AXIS)
    # -----------------------------

    """
    We define the state vector for ONE AXIS (x-direction) as:

        x1 = x              (table position)
        x2 = x_dot          (table velocity)
        x3 = alpha          (pencil angle)
        x4 = alpha_dot      (pencil angular velocity)

    So the state vector is:

        X = [ x, x_dot, alpha, alpha_dot ]^T

    The input is:

        u = x_des

    Goal:
    Convert second-order equations into first-order form:

        X_dot = A_x * X + B_x * u
    """


    # -----------------------------
    # ORIGINAL SECOND-ORDER EQUATIONS
    # -----------------------------

    """
    Table dynamics:

        tau^2 * x_ddot + 2*zeta*tau * x_dot + x = u

    Rewritten:

        x_ddot = (1/tau^2)*(u - x) - (2*zeta/tau)*x_dot

    Pencil dynamics:

        alpha_ddot = (g/l)*alpha - (1/l)*x_ddot

    Substitute x_ddot into pencil equation:

        alpha_ddot =
            (g/l)*alpha
            - (1/l)*[(1/tau^2)*(u - x) - (2*zeta/tau)*x_dot]
    """


    # -----------------------------
    # CONVERT TO FIRST-ORDER FORM
    # -----------------------------

    """
    First-order equations:

    1) x1_dot = x2

    2) x2_dot =
        (1/tau^2)*(u - x1)
        - (2*zeta/tau)*x2

    3) x3_dot = x4

    4) x4_dot =
        (g/l)*x3
        - (1/l)*[(1/tau^2)*(u - x1)
                    - (2*zeta/tau)*x2]

    Now we collect terms in form:

        X_dot = A_x * X + B_x * u
    """


    # -----------------------------
    # BUILD A_x MATRIX (4x4, ONE AXIS)
    # -----------------------------

    A_x = np.array([
        [0, 1, 0, 0],
        [-1/tau**2, -2*zeta/tau, 0, 0],
        [0, 0, 0, 1],
        [1/(l*tau**2), 2*zeta/(l*tau), g/l, 0]
    ])


    # -----------------------------
    # BUILD B_x VECTOR (4x1, ONE AXIS)
    # -----------------------------

    B_x = np.array([
        [0],
        [1/tau**2],
        [0],
        [-1/(l*tau**2)]
    ])


    # -----------------------------
    # BUILD FULL 2D SYSTEM (8 STATES, 2 INPUTS)
    # -----------------------------

    """
    Since x and y directions are decoupled in our simplified model,
    the total A matrix is block diagonal:

            [ A_x   0  ]
        A = [  0   A_y ]

    Where A_y = A_x (same dynamics)

    Similarly, B becomes:

            [ B_x   0  ]
        B = [  0   B_y ]

    Inputs:
        u = [x_des, y_des]^T
    """

    # Zero blocks
    Z4 = np.zeros((4, 4))
    Z4x1 = np.zeros((4, 1))

    # Full A (8x8)
    A = np.block([
        [A_x, Z4],
        [Z4,  A_x]
    ])

    # Full B (8x2)
    B = np.block([
        [B_x, Z4x1],
        [Z4x1, B_x]
    ])
    
    return A, B


# -----------------------------
# Main Simulation Entry Point
# -----------------------------
if __name__ == "__main__":

    plant = PencilPlant()
    
    A, B = BuildLinearModel()
    Crank = np.linalg.matrix_rank(ct.ctrb(A, B))
    print(f"Is the system controllable (full rank)? {Crank==num_states}")    
    
    # -----------------------------
    # We have two ways to find K, either placing poles:
    
    desired_poles = [
        -8, -10, -12, -14,   # x-axis
        -8, -10, -12, -14    # y-axis
    ]
    
    K = ct.place(A, B, desired_poles)
    
    
    # -----------------------------
    # Or we use LQR:
    
    #K, S, E = ct.lqr(A, B, Q, R)
    
    
    # Now we use that K for our simple controller
    controller = StateFeedbackController(K)

    sim = Simulator(
        plant=plant,
        controller=controller,
        dt=0.001
    )

    total_time = 3.0
    steps = int(total_time / sim.dt)


    # RUN SIMULATION

    history = np.zeros((steps, 8))

    print("Running simulation...")

    for i in range(steps):
        state = sim.step()
        history[i, :] = state.as_vector()

        if abs(state.alpha_x) > 0.5 or abs(state.alpha_y) > 0.5:
            print(f"Pencil fell at step {i}")
            history = history[:i+1]
            break

    print("Simulation complete.")

    # SAVE TO simulations/ FOLDER
    
    '''
    os.makedirs("simulations", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulations/pencil_sim_{timestamp}.npz"

    np.savez(
        filename,
        history=history,
        dt=sim.dt
    )

    print(f"Simulation saved to: {filename}")
    
    '''
    
    # Load simulation
    
    '''
    
    data = np.load("simulations/pencil_sim_20260226_231455.npz")
    history = data["history"]
    dt = data["dt"]
    
    '''

    viz = Visualizer3D(history, sim.dt)
    viz.render_video(video_speed=1)   # 5x slower than real time
    
    