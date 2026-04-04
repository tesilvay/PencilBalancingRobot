from src.shared import State, Measurement, ControlInput
import numpy as np


class BaseEstimator:

    def estimate(
        self, 
        y: Measurement, 
        dt: float, 
        u_cmd: ControlInput
    ) -> tuple[State, np.ndarray]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def calc_innovation(self, y_meas: Measurement, x: State) -> np.ndarray:
        # H picks [px, ax, py, ay] from state — same semantics as Kalman H
        x_hat_meas = np.array([x.px, x.ax, x.py, x.ay])
        return y_meas.as_vector() - x_hat_meas
        




