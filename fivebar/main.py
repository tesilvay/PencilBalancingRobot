from dataclasses import dataclass
from transform import FiveBarTransform
from mechanism import FiveBarMechanism
from workspace import FiveBarWorkspace
from visualization import FiveBarVisualizer

MIN_ANGLE_DEG = 20.0  # Minimum angle (deg) away from straight/collinear; tune for safe workspace margin.


@dataclass
class MechanismParams:
    """Five-bar geometry (mm)."""
    O: tuple[float, float]
    B: tuple[float, float]
    la: float
    lb: float
    
if __name__ == "__main__":

    mechanism=MechanismParams(
                O=(239, 288),
                B=(212, 320),
                la=170,
                lb=170,
            )


    tf = FiveBarTransform(mechanism.O, mechanism.B)

    mech = FiveBarMechanism(tf, la=mechanism.la, lb=mechanism.lb, min_angle_deg=MIN_ANGLE_DEG)

    workspace = FiveBarWorkspace(mech)

    points = workspace.sweep_cartesian(70)

    viz = FiveBarVisualizer(mech, workspace)

    viz.interactive_workspace(points)