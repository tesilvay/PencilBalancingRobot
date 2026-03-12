from dataclasses import dataclass
from transform import FiveBarTransform
from mechanism import FiveBarMechanism
from workspace import FiveBarWorkspace
from visualization import FiveBarVisualizer

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

    mech = FiveBarMechanism(tf, la=mechanism.la, lb=mechanism.lb)

    workspace = FiveBarWorkspace(mech)

    angles, points = workspace.sweep_joint_space(100)

    viz = FiveBarVisualizer(mech, workspace)

    viz.interactive_workspace(points)