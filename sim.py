from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 2D line as seen by a camera
# -----------------------------

@dataclass
class LineObservation:
    slope: float      # s
    offset: float     # b (z_intercept)


# -----------------------------
# Camera geometry
# -----------------------------

@dataclass
class CameraPosition:
    x: float
    y: float


# -----------------------------
# Reconstructed 3D pencil state
# -----------------------------

@dataclass
class PencilState:
    X: float          # position at camera height
    Y: float
    alpha_x: float    # tilt in x-direction
    alpha_y: float    # tilt in y-direction


# -----------------------------
# Reconstruction engine
# -----------------------------

class PencilReconstructor:

    def __init__(self, cam1_pos: CameraPosition, cam2_pos: CameraPosition):
        self.cam1 = cam1_pos
        self.cam2 = cam2_pos

    def reconstruct(
        self,
        cam1_obs: LineObservation,
        cam2_obs: LineObservation
    ) -> PencilState:

        b1 = cam1_obs.offset
        s1 = cam1_obs.slope

        b2 = cam2_obs.offset
        s2 = cam2_obs.slope

        xr = self.cam2.x
        yr = self.cam1.y

        denom = b1 * b2 + 1.0

        if abs(denom) < 1e-8:
            raise ValueError("Degenerate configuration: b1 * b2 + 1 ≈ 0")

        X = (b1 * yr + b1 * b2 * xr) / denom
        Y = (b2 * xr - b1 * b2 * yr) / denom

        alpha_x = (s1 + b1 * s2) / denom
        alpha_y = (s2 - b2 * s1) / denom

        return PencilState(X, Y, alpha_x, alpha_y)
    

# -----------------------------
# Filters
# -----------------------------

class LowPassFilter:
    def __init__(self, smoothing_factor: float):
        if not 0 < smoothing_factor <= 1: # 0 is very smooth / 1 means no filter
            raise ValueError("smoothing_factor must be in (0,1]")
        self.smoothing_factor = smoothing_factor
        self.state = None

    def update(self, measurement: float) -> float:
        if self.state is None:
            self.state = measurement
        else:
            self.state = (
                self.smoothing_factor * measurement
                + (1 - self.smoothing_factor) * self.state
            )
        return self.state
    

# -----------------------------
# Pure P Controller (position + slope only)
# -----------------------------
    
class PController2D:

    def __init__(self, g_position: float, g_alpha: float):
        self.g_position = g_position
        self.g_alpha = g_alpha

    def compute(self, state: PencilState) -> tuple[float, float]:

        x_des = self.g_position * state.X + self.g_alpha * state.alpha_x
        y_des = self.g_position * state.Y + self.g_alpha * state.alpha_y

        return x_des, y_des


class TableController:

    def __init__(
        self,
        controller: PController2D,
        filter_smoothing_factor: float = 0.2
    ):
        self.controller = controller

        # One LPF per state variable
        self.f_X = LowPassFilter(filter_smoothing_factor)
        self.f_Y = LowPassFilter(filter_smoothing_factor)
        self.f_alpha_x = LowPassFilter(filter_smoothing_factor)
        self.f_alpha_y = LowPassFilter(filter_smoothing_factor)

    def update(self, raw_state: PencilState) -> tuple[float, float]:

        filtered = PencilState(
            X=self.f_X.update(raw_state.X),
            Y=self.f_Y.update(raw_state.Y),
            alpha_x=self.f_alpha_x.update(raw_state.alpha_x),
            alpha_y=self.f_alpha_y.update(raw_state.alpha_y),
        )

        return self.controller.compute(filtered)
    


# -----------------------------
# Simulation Plant
# -----------------------------

class PencilPlant:

    def __init__(self, L=0.2):
        self.g = 9.81
        self.l = L / 2

        # pencil states
        self.alpha_x = 0.3  # small initial tilt
        self.alpha_y = -0.1
        self.alpha_x_dot = 0.0
        self.alpha_y_dot = 0.0

        # table states
        self.x = 0.0
        self.y = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0

        # table dynamics params
        self.tau = 0.04
        self.zeta = 0.7

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
  
        # position of pencil at camera height
        X = self.x + self.l * self.alpha_x
        Y = self.y + self.l * self.alpha_y

        return X, Y, self.alpha_x, self.alpha_y
    
    
# -----------------------------
# Control Loop
# -----------------------------

class ControlLoop:

    def __init__(
        self,
        reconstructor: PencilReconstructor,
        table_controller: TableController,
        loop_dt: float = 0.002  # 2 ms loop (500 Hz)
    ):
        self.reconstructor = reconstructor
        self.table_controller = table_controller
        self.loop_dt = loop_dt
        self.running = False

    def step(
        self,
        cam1_obs: LineObservation,
        cam2_obs: LineObservation
    ) -> tuple[float, float]:

        # 1. Reconstruct 3D pencil state
        pencil_state = self.reconstructor.reconstruct(cam1_obs, cam2_obs)

        # 2. Compute desired table position
        x_des, y_des = self.table_controller.update(pencil_state)

        # 3. Assume table moves instantly
        return x_des, y_des

    def run(self, observation_source):

        self.running = True

        while self.running:

            cam1_obs, cam2_obs = observation_source()

            x_des, y_des = self.step(cam1_obs, cam2_obs)

            # Here x_des and y_des go to the hardware
            # For now we just print
            print(f"Commanded table position: {x_des:.3f}, {y_des:.3f}")

            time.sleep(self.loop_dt)

    def stop(self):
        self.running = False
        
        
# -----------------------------
# Simulation
# -----------------------------
class Simulator:

    def __init__(self, plant, table_controller, dt=0.001):
        self.plant = plant
        self.controller = table_controller
        self.dt = dt

        self.x_des = 0.0
        self.y_des = 0.0

    def step(self):

        # 1. advance physical world
        X, Y, alpha_x, alpha_y = self.plant.step(
            self.x_des,
            self.y_des,
            self.dt
        )

        # 2. build "measured" state (ground truth for now)
        state = PencilState(
            X=X,
            Y=Y,
            alpha_x=alpha_x,
            alpha_y=alpha_y
        )

        # 3. controller computes new command
        self.x_des, self.y_des = self.controller.update(state)

        return state


# -----------------------------
# 3D Visualization
# -----------------------------

class Visualizer3D:

    def __init__(self, plant, L=0.2):

        self.plant = plant
        self.L = L

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(-0.15, 0.15)
        self.ax.set_ylim(-0.15, 0.15)
        self.ax.set_zlim(0, L)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Table square (will update)
        self.table_plot, = self.ax.plot([], [], [], 'k-', linewidth=2)

        # Pencil line
        self.pencil_plot, = self.ax.plot([], [], [], 'r-', linewidth=3)
        
        # text for the pencil state
        self.info_text = self.ax.text2D(
            0.02, 0.95,
            "",
            transform=self.ax.transAxes
        )

        plt.ion()
        plt.show()

    def update(self):

        x = self.plant.x
        y = self.plant.y
        alpha_x = self.plant.alpha_x
        alpha_y = self.plant.alpha_y

        # --- Table corners ---
        size = 0.05

        corners_x = [
            x - size, x + size,
            x + size, x - size,
            x - size
        ]

        corners_y = [
            y - size, y - size,
            y + size, y + size,
            y - size
        ]

        corners_z = [0, 0, 0, 0, 0]

        self.table_plot.set_data(corners_x, corners_y)
        self.table_plot.set_3d_properties(corners_z)

        # --- Pencil line ---
        base = np.array([x, y, 0])

        direction = np.array([
            alpha_x,
            alpha_y,
            1.0
        ])

        direction = direction / np.linalg.norm(direction)

        tip = base + direction * self.L

        self.pencil_plot.set_data(
            [base[0], tip[0]],
            [base[1], tip[1]]
        )

        self.pencil_plot.set_3d_properties(
            [base[2], tip[2]]
        )
        
        # Pencil state text on the sim
        info_string = (
            f"X: {self.plant.x:.3f}\n"
            f"Y: {self.plant.y:.3f}\n"
            f"αx: {self.plant.alpha_x:.3f}\n"
            f"αy: {self.plant.alpha_y:.3f}"
        )

        self.info_text.set_text(info_string)

        plt.draw()
        plt.pause(0.001)
    
# -----------------------------
# Main Simulation Entry Point
# -----------------------------
if __name__ == "__main__":

    plant = PencilPlant(L=0.2)

    p_controller = PController2D(
        g_position=2.5,
        g_alpha=50.0
    )

    table_controller = TableController(
        controller=p_controller,
        filter_smoothing_factor=0.95
    )

    sim = Simulator(
        plant=plant,
        table_controller=table_controller,
        dt=0.001
    )

    viz = Visualizer3D(plant, L=0.2)

    total_time = 10.0
    steps = int(total_time / sim.dt)
    
    speed = 1  # 25% real time (slow motion)

    try:
        for i in range(steps):
            
            loop_start = time.time()

            if not plt.fignum_exists(viz.fig.number):
                print("Window closed. Stopping simulation.")
                break

            state = sim.step()

            if abs(state.alpha_x) > 0.5 or abs(state.alpha_y) > 0.5:
                print("Pencil fell.")
                break

            viz.update()

            # ---- slow-down control ----
            elapsed = time.time() - loop_start
            desired_duration = sim.dt / speed
            
            time.sleep(max(0, desired_duration - elapsed))

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")