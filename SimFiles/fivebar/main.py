from transform import FiveBarTransform
from mechanism import FiveBarMechanism
from workspace import FiveBarWorkspace
from visualization import FiveBarVisualizer


if __name__ == "__main__":

    O = [83,57]
    B = [61,88]

    tf = FiveBarTransform(O,B)

    mech = FiveBarMechanism(tf, la=77, lb=69.6)

    workspace = FiveBarWorkspace(mech)

    angles, points = workspace.sweep_joint_space(200)

    viz = FiveBarVisualizer(mech, workspace)

    viz.interactive_workspace(points)