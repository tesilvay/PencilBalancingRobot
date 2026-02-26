import numpy as np
from dataclasses import dataclass
import time


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

        x_desired = self.g_position * state.X + self.g_alpha * state.alpha_x
        y_desired = self.g_position * state.Y + self.g_alpha * state.alpha_y

        return x_desired, y_desired


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
        x_desired, y_desired = self.table_controller.update(pencil_state)

        # 3. Assume table moves instantly
        return x_desired, y_desired

    def run(self, observation_source):

        self.running = True

        while self.running:

            cam1_obs, cam2_obs = observation_source()

            x_desired, y_desired = self.step(cam1_obs, cam2_obs)

            # Here x_desired and y_desired go to the hardware
            # For now we just print
            print(f"Commanded table position: {x_desired:.3f}, {y_desired:.3f}")

            time.sleep(self.loop_dt)

    def stop(self):
        self.running = False