from dataclasses import dataclass
import os
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
        la=175,
        lb=175,
    )


    tf = FiveBarTransform(mechanism.O, mechanism.B)

    mech = FiveBarMechanism(tf, la=mechanism.la, lb=mechanism.lb, min_angle_deg=MIN_ANGLE_DEG)

    workspace = FiveBarWorkspace(mech)

    MAX_RES = 100
    MIN_RES = 20
    ALPHA = 0.01  # use heuristic inside compare_adaptive_to_full

    # Optional: Numba vs Python timing comparison (set COMPARE_NUMBA=1).
    if os.environ.get("COMPARE_NUMBA") == "1":
        # With Numba (default)
        os.environ.pop("USE_NUMBA", None)
        t0 = time.perf_counter()
        pts_full_numba = workspace.sweep_cartesian_full(MAX_RES)
        t_full_numba = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        pts_adapt_numba = workspace.sweep_cartesian_adaptive(
            max_res=MAX_RES, min_res=MIN_RES, samples_per_cell=2
        )
        t_adapt_numba = (time.perf_counter() - t1) * 1000.0

        # Without Numba
        os.environ["USE_NUMBA"] = "0"
        t2 = time.perf_counter()
        pts_full_py = workspace.sweep_cartesian_full(MAX_RES)
        t_full_py = (time.perf_counter() - t2) * 1000.0
        t3 = time.perf_counter()
        pts_adapt_py = workspace.sweep_cartesian_adaptive(
            max_res=MAX_RES, min_res=MIN_RES, samples_per_cell=2
        )
        t_adapt_py = (time.perf_counter() - t3) * 1000.0
        os.environ.pop("USE_NUMBA", None)

        def point_set_same(a, b):
            if a.shape[0] != b.shape[0]:
                return False
            sa = set(tuple(p) for p in a)
            sb = set(tuple(p) for p in b)
            return sa == sb

        print("--- Numba vs Python comparison ---")
        print(f"Full sweep:    Numba {t_full_numba:.1f} ms, {pts_full_numba.shape[0]} pts  |  Python {t_full_py:.1f} ms, {pts_full_py.shape[0]} pts")
        print(f"Adaptive sweep: Numba {t_adapt_numba:.1f} ms, {pts_adapt_numba.shape[0]} pts  |  Python {t_adapt_py:.1f} ms, {pts_adapt_py.shape[0]} pts")
        print(f"Full point sets match: {point_set_same(pts_full_numba, pts_full_py)}")
        print(f"Adaptive point sets match: {point_set_same(pts_adapt_numba, pts_adapt_py)}")
        print("---")
        pts_full = pts_full_numba
        pts_adapt = pts_adapt_numba
    else:
        result = workspace.compare_adaptive_to_full(
            max_res=MAX_RES,
            alpha=ALPHA,
            min_res=MIN_RES,
            samples_per_cell=2,
        )
        pts_full = result["pts_full"]
        pts_adapt = result["pts_adapt"]
        print(f"Full sweep:     {pts_full.shape[0]} valid points, {result['time_full_ms']:.1f} ms")
        print(f"Adaptive sweep: {pts_adapt.shape[0]} valid points, {result['time_adaptive_ms']:.1f} ms")
        print(
            "Geometric error:"
            f" area_rel_error={result['area_rel_error']:.3f},"
            f" sym_diff_ratio={result['sym_diff_ratio']:.3f},"
            f" hausdorff_distance={result['hausdorff_distance']:.3f}"
        )

    viz = (mech, workspace)
    viz.interactive_workspace(pts_adapt)