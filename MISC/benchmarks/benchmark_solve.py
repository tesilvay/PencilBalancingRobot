"""
Benchmark mech.solve(): Numba path vs Python path.

Run from repo root:
  PYTHONPATH=. python benchmarks/benchmark_solve.py

Reports time for many solve() calls with Numba enabled and with Numba disabled,
and the speedup when Numba is installed.
"""

import time

import numpy as np

from fivebar.transform import FiveBarTransform
from fivebar.mechanism import FiveBarMechanism
from fivebar import numba_solve


def get_mech():
    """Same mechanism geometry as main/system_builder."""
    tf = FiveBarTransform((128.77, 178.13), (101.77, 210.13))
    return FiveBarMechanism(tf, la=175, lb=175)


def get_valid_target(mech):
    """One target in mm (global) that solve() accepts."""
    theta1, theta4 = 0.9, 2.3
    A_l, C_l, P1_l, P2_l = mech.fk(theta1, theta4)
    P_l = P1_l if P1_l[1] > 0 else P2_l
    return np.array([float(mech.tf.l2g(P_l)[0]), float(mech.tf.l2g(P_l)[1])])


def run_benchmark(n_calls=50_000):
    mech = get_mech()
    target = get_valid_target(mech)

    # Warm up (JIT compile if Numba)
    for _ in range(200):
        mech.solve(target)

    # --- With Numba (or Python if Numba not installed) ---
    t0 = time.perf_counter()
    for _ in range(n_calls):
        mech.solve(target)
    t_numba_path = time.perf_counter() - t0

    # --- Without Numba: force Python path ---
    had_numba = numba_solve.HAS_NUMBA
    numba_solve.HAS_NUMBA = False
    mech._numba_constants = None

    t0 = time.perf_counter()
    for _ in range(n_calls):
        mech.solve(target)
    t_python_path = time.perf_counter() - t0

    numba_solve.HAS_NUMBA = had_numba
    mech._numba_constants = None

    return t_numba_path, t_python_path, had_numba


def main():
    n = 50_000
    print("Solve benchmark (five-bar mech.solve)")
    print(f"  Calls per path: {n}")
    t_numba, t_python, had_numba = run_benchmark(n)

    us_numba = 1e6 * t_numba / n
    us_python = 1e6 * t_python / n
    print(f"  Numba path:   {t_numba:.4f} s  ({us_numba:.2f} µs/call)")
    print(f"  Python path:  {t_python:.4f} s  ({us_python:.2f} µs/call)")

    if had_numba and t_numba > 0:
        speedup = t_python / t_numba
        print(f"  Speedup:      {speedup:.2f}x faster with Numba")
    else:
        print("  (Numba not installed; both runs used Python path.)")


if __name__ == "__main__":
    main()
