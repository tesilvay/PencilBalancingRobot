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

    def set_workspace_offset(self, dx: float, dy: float) -> None:
        self.workspace_offset = (float(dx), float(dy))

    def command_to_angles(self, command) -> tuple[float, float]:
        x = command.px_cmd + self.workspace_offset[0]
        y = command.py_cmd + self.workspace_offset[1]
        target_mm = np.array([x, y]) * 1000.0
        theta1, theta2 = self._mech.ik(target_mm)
        return np.rad2deg(theta1), np.rad2deg(theta2)

MECHANISM_REGISTRY = {
    "five_bar": Spec(Mechanism, MechanismParams, MECHANISM_PRESETS),
}