from dataclasses import dataclass
import time
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

    mechanism = MechanismParams(
        O=(239, 288),
        B=(212, 320),
        la=170,
        lb=120,
    )


    tf = FiveBarTransform(mechanism.O, mechanism.B)

    mech = FiveBarMechanism(tf, la=mechanism.la, lb=mechanism.lb, min_angle_deg=MIN_ANGLE_DEG)

    workspace = FiveBarWorkspace(mech)

    MAX_RES = 70
    MIN_RES = 5
    ALPHA = 0.05  # use heuristic inside compare_adaptive_to_full

    result = workspace.compare_adaptive_to_full(
        max_res=MAX_RES,
        alpha=ALPHA,
        min_res=MIN_RES,
        samples_per_cell=3,
    )

    pts_full = result["pts_full"]
    pts_adapt = result["pts_adapt"]

    print(f"Full sweep:     {pts_full.shape[0]} valid points, {result['time_full_ms']:.1f} ms")
    print(f"Adaptive sweep: {pts_adapt.shape[0]} valid points, {result['time_adaptive_ms']:.1f} ms")
    print(
        "Geometric error:"
        f" area_rel_error={result['area_rel_error']},"
        f" sym_diff_ratio={result['sym_diff_ratio']},"
        f" hausdorff_distance={result['hausdorff_distance']}"
    )

    viz = FiveBarVisualizer(mech, workspace)

    # Visualize adaptive sweep by default; switch to pts_full for ground truth.
    viz.interactive_workspace(pts_adapt)