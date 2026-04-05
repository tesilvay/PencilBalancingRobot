from dataclasses import dataclass
import numpy as np
from src.shared import Spec
from src.system.actuator.mech.transform import FiveBarTransform
from src.system.actuator.mech.mechanism import FiveBarMechanism

# WARNING
# ALL VALUES ARE IN MM FOR THIS CLASS

@dataclass
class MechanismParams:
    O:  np.ndarray   # origin point
    B:  np.ndarray   # base point
    la: float        # link a length
    lb: float        # link b length

MECHANISM_PRESETS = {
    "default": {
        "O":  np.array([128.77, 178.13]),
        "B":  np.array([101.77, 210.13]),
        "la": 175,
        "lb": 175,
    }
}

class Mechanism:
    def __init__(self, params: MechanismParams):
        tf = FiveBarTransform(params.O, params.B)
        self._mech = FiveBarMechanism(tf, la=params.la, lb=params.lb)
        self.workspace_offset = (0.0, 0.0)

    @property
    def tf(self):
        """Transform shared with :class:`FiveBarMechanism` (O_g, B_g, …)."""
        return self._mech.tf

    def set_workspace_offset(self, dx: float, dy: float) -> None:
        self.workspace_offset = (float(dx), float(dy))

    def command_geometry(
        self, command
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Single IK+FK solve for the table command (meters → mm internally).

        Returns
        -------
        joints_mm : ndarray, shape (3, 2)
            Rows: A, C, P in global mm.
        (theta1_deg, theta4_deg) : tuple[float, float]
            Servo angles in degrees (same convention as :meth:`command_to_angles`).
        """
        x = float(command.px_cmd) + self.workspace_offset[0]
        y = float(command.py_cmd) + self.workspace_offset[1]
        target_mm = np.array([x, y], dtype=float) * 1000.0
        theta1, theta4, A_g, C_g, P_g = self._mech.solve(target_mm)
        joints = np.stack(
            [
                np.asarray(A_g, dtype=float).reshape(-1)[:2],
                np.asarray(C_g, dtype=float).reshape(-1)[:2],
                np.asarray(P_g, dtype=float).reshape(-1)[:2],
            ],
            axis=0,
        )
        theta_deg = (float(np.rad2deg(theta1)), float(np.rad2deg(theta4)))
        return joints, theta_deg

    def command_to_angles(self, command) -> tuple[float, float]:
        _, theta_deg = self.command_geometry(command)
        return theta_deg

MECHANISM_REGISTRY = {
    "five_bar": Spec(Mechanism, MechanismParams, MECHANISM_PRESETS),
}