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
                O=(85.91, 57.86),
                B=(60.10, 87.07),
                la=76.84,
                lb=66.83,
            )


    tf = FiveBarTransform(mechanism.O, mechanism.B)

    mech = FiveBarMechanism(tf, la=mechanism.la, lb=mechanism.lb)

    workspace = FiveBarWorkspace(mech)

    angles, points = workspace.sweep_joint_space(200)

    viz = FiveBarVisualizer(mech, workspace)

    viz.interactive_workspace(points)