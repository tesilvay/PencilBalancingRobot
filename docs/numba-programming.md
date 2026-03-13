# Numba programming guide

This guide explains how Numba was used in the fivebar workspace sweeps, what went wrong along the way, and how to apply the same ideas elsewhere.

---

## 1. What is Numba?

Numba is a JIT (just-in-time) compiler for Python. It compiles a subset of Python and NumPy into fast machine code (via LLVM). You decorate functions with `@numba.njit` (or `@numba.jit(nopython=True)`); when they are first called, Numba compiles them, and later calls run at near-C speed.

- **Good for:** Tight numeric loops, scalar math, small arrays, no Python objects.
- **Not for:** Code that uses classes, exceptions, dynamic types, or external libraries (e.g. Shapely) inside the compiled region.

---

## 2. What Numba can and cannot speed up (in this project)

### What we accelerated

The **per-point work** that runs thousands of times in a sweep:

- **Reachability check:** Transform global → local (`g2l`), two distance checks vs `r_min`/`r_max`.
- **Inverse kinematics (IK):** Scalar formulas with `sqrt`, `arctan2`, discriminants.
- **Forward kinematics (FK):** `cos`/`sin`, vector math, norms, branch choice.
- **Valid-config checks:** Cross products, norms, comparisons (cranks, elbows, coupler, point-in-front).

All of that is pure floats and simple math, so it lives in a single `@numba.njit` function that returns `True`/`False`. Each call is then a fast C-level function instead of many Python steps and exception checks.

### What stays in Python (not accelerated)

- The **outer loops** over grid points or quadtree cells (`for x in xs`, `while queue`, etc.).
- **List appends**, `deque`, `tqdm`, `np.unique`, building the final point array.
- Any code that calls `mech.solve`, Shapely, or raises/catches exceptions.

So the gain is: the **inner “is this point valid?”** work is compiled; the **structure** of the sweep stays in Python. Because the inner work dominated runtime, both full and adaptive sweeps got a large speedup (e.g. full ~2.7×, adaptive ~25×) once we fixed the pitfalls below.

---

## 3. How we used Numba in the fivebar workspace

### 3.1 Design

- **One kernel function:** `point_valid_numba(x_g, y_g, o_g_0, o_g_1, rt_00, rt_01, rt_10, rt_11, lc, la, lb, r_min, r_max, min_sin)` → `bool`.
- **Constants come from Python:** The mechanism and transform are used once to build a flat set of scalars (`get_numba_constants(mech)`). Those are passed into every call so the JIT function stays pure (no classes, no NumPy arrays in the signature if we can avoid it).
- **No exceptions inside the kernel:** Instead of `raise ValueError(...)`, we `return False`. The Python side only needs to know “valid or not” for the sweep.

### 3.2 Passing constants: scalars, not arrays

**Mistake we avoided:** Passing `O_g` as a length-2 array and `R_T` as a 2×2 array into `@numba.njit` can work, but Numba’s type inference and ABI are simpler and more robust when the signature uses **scalar floats** for everything.

So we “flattened” the constants:

- `O_g` → `o_g_0`, `o_g_1`
- `R_T` (2×2) → `rt_00`, `rt_01`, `rt_10`, `rt_11`
- Plus scalars: `lc`, `la`, `lb`, `r_min`, `r_max`, `min_sin`

The kernel then uses only scalar arithmetic; no array indexing inside the JIT. The Python side builds a dict with these keys and calls `point_valid_numba(x, y, **nc)`.

### 3.3 Use `math` inside the kernel, not `numpy`

**What we did:** Inside the `@numba.njit` function we use `math.sqrt`, `math.atan2`, `math.cos`, `math.sin`, `math.fabs`, not `np.sqrt`, etc.

**Why:** Numba compiles the `math` module’s functions very well and avoids pulling in NumPy’s dispatching. For scalar floats, `math` is the right choice and avoids subtle typing/overload issues.

### 3.4 No helper functions from Python inside the JIT

**Problem:** We had a small 2D cross-product helper `_cross2(ax, ay, bx, by)`. We wanted to call it from inside `point_valid_numba`.

**Error:** You cannot call a **non-JIT** Python function from inside an `@numba.njit` function. Numba would try to compile the call and fail or fall back to object mode (slow).

**Fix:** We **inlined** the 2D cross product as `ax * by - ay * bx` everywhere it was needed inside the kernel, and removed the separate helper from the JIT path. No Python calls inside the compiled function.

### 3.5 Graceful fallback when Numba is not installed

**Requirement:** The code must run even if the user has not installed Numba (e.g. `pip install numba` not run).

**Approach:**

- At import time: `try: import numba; HAS_NUMBA = True` / `except ImportError: HAS_NUMBA = False`.
- If `HAS_NUMBA` is False, we define a **stub** `point_valid_numba(...)` that raises `RuntimeError("Numba is not installed; use Python sweep path.")`.
- In the workspace, we only use the Numba path when `HAS_NUMBA and point_valid_numba is not None and os.environ.get("USE_NUMBA", "1") != "0"`. Otherwise we use the original `reachable_fast` + `mech.solve` path.

So “no Numba” is a simple, explicit fallback instead of a crash at import or at first call.

### 3.6 Optional disable for comparison: `USE_NUMBA`

We wanted to **compare** “same sweep with Numba vs without” (same point set, different timings). So we added an environment variable:

- `USE_NUMBA=0` (or unset and we force it in code when doing Python-side comparison): use the Python `mech.solve` path.
- Otherwise: use the Numba kernel when available.

The workspace checks `os.environ.get("USE_NUMBA", "1") != "0"` when deciding which path to use. In `main.py`, when we run the “Python” leg of a comparison, we set `os.environ["USE_NUMBA"] = "0"` before that sweep and pop it after, so the rest of the run still uses Numba by default.

---

## 4. The “Numba was slower” bug: JIT warm-up

### What you saw

- **Single adaptive sweep:** e.g. “Adaptive sweep: 641 valid points, **424.2 ms**” and a low cell/s rate (~3700 cells/s).
- **Compare full vs adaptive:** Full sweep ~411 ms, then “Adaptive sweep: 641 valid points, **25.1 ms**” and a much higher rate (~66652 cells/s).

So in one case “Numba” looked **much slower** for the same adaptive sweep.

### Root cause

Numba compiles the JIT function **on first use**. That first call pays a one-time cost (hundreds of milliseconds) for compilation. After that, the function is cached and every later call is fast.

- **Single adaptive run:** The **first** call to `point_valid_numba` in the whole process happened inside the adaptive sweep. So the 424 ms included **JIT compile + sweep**. The timer in `main.py` was around the whole `sweep_cartesian_adaptive(...)` call, so it was correct but misleading: it was not “sweep only.”
- **Compare run:** The **full** sweep ran first, so the first call to `point_valid_numba` happened there. By the time the **adaptive** sweep ran, the kernel was already compiled. So the adaptive timing was only the real sweep (~25 ms), and the full sweep timing included the one-time compile.

So Numba was **not** slower; the single adaptive run was the one that paid the compile cost, and we were timing “compile + sweep” instead of “sweep only.”

### Fix: warm-up before timing

We did two things:

1. **Explicit warm-up in the single-run path**  
   In `main.py`, when we run a **single** full or adaptive sweep, we call `workspace.warm_up_numba()` **before** starting the timer. `warm_up_numba()` does a single call to `point_valid_numba` (e.g. at the bounds corner) so that the first compilation happens **outside** the timed region. The reported time is then “sweep only” and matches the ~25 ms you see for adaptive in the compare run.

2. **Warm-up inside each sweep (optional but consistent)**  
   At the start of `sweep_cartesian_full` and `sweep_cartesian_adaptive`, when using Numba we also do one dummy call to `point_valid_numba` before the main loop. So whichever sweep runs first in a “compare” run compiles the kernel; the second sweep then runs with no compile cost. The first sweep’s time still includes compile, but that’s acceptable when comparing full vs adaptive.

**Lesson:** For any Numba kernel used in a **timed** code path, either:

- Time only **after** a warm-up call, or  
- Clearly document that the **first** run includes JIT compile and is not representative of steady-state performance.

---

## 5. File layout and wiring

- **`fivebar/numba_solve.py`**
  - `HAS_NUMBA`, `point_valid_numba`, `get_numba_constants(mech)`.
  - Kernel implements: g2l, reachability, IK, FK, branch choice, valid_config (cranks, elbows, coupler, point-in-front), all with scalars and `math.*`.
- **`fivebar/workspace.py`**
  - Imports `HAS_NUMBA`, `get_numba_constants`, `point_valid_numba` (with fallback if the module is missing).
  - `_get_numba_constants()`: caches the constant dict per workspace.
  - `warm_up_numba()`: one `point_valid_numba(...)` call to trigger JIT.
  - In `sweep_cartesian_full`, `sweep_cartesian_adaptive`, and `_augment_points_on_axes`: when `use_numba` is True, the inner “is this point valid?” is `point_valid_numba(x, y, **nc)` (or axis variants); otherwise the original `reachable_fast` + `mech.solve` path is used.

### Import fallback

When the project is run as a script (e.g. `python fivebar/main.py`), the package might not be a package, so `from .numba_solve import ...` can fail. We use:

```python
try:
    from .numba_solve import HAS_NUMBA, get_numba_constants, point_valid_numba
except ImportError:
    try:
        from numba_solve import HAS_NUMBA, get_numba_constants, point_valid_numba
    except ImportError:
        HAS_NUMBA = False
        get_numba_constants = None
        point_valid_numba = None
```

So it works both as a package (`python -m fivebar.main`) and when running from the `fivebar` directory.

---

## 6. How to run and compare

In `fivebar/main.py` we have:

- **`COMPARE_NUMBA`:** If `True`, we run the same sweep(s) with Numba and with Python (`USE_NUMBA=0`) and print timings and point-set equality.
- **`COMPARE_SWEEP`:** If `True`, we run both full and adaptive and (when applicable) report geometric metrics.
- **`PREFER_ALGO`:** `"full"` or `"adaptive"` for single-sweep mode or for “which algo to compare Numba vs Python” when not comparing sweeps.

For a **single** run (no comparisons), we call `workspace.warm_up_numba()` before starting the timer so the reported time is sweep-only and Numba is used by default.

---

## 7. Summary: lessons and checklist

| Issue | What happened | Fix / lesson |
|--------|----------------|--------------|
| **Numba “slower” in one run** | First call to the kernel paid JIT compile; we timed “compile + sweep” for single adaptive. | Warm-up call **before** the timer so reported time is “sweep only.” |
| **Calling Python from JIT** | Using a small helper (e.g. cross product) from inside `@numba.njit` fails or forces object mode. | Inline the logic or use a `@numba.njit` helper; no non-JIT calls inside the compiled function. |
| **Constants in the kernel** | Passing arrays can work but complicates typing. | Flatten to scalar arguments (e.g. `o_g_0`, `o_g_1`, `rt_00`, …) and pass a dict from Python with `**nc`. |
| **NumPy vs math** | Using `np.sqrt`, `np.atan2` etc. inside njit is fine but can add overhead. | Prefer `math.sqrt`, `math.atan2`, etc. for scalar math in the kernel. |
| **No Numba installed** | Import or first call would crash. | `try/except ImportError`, set `HAS_NUMBA`; provide a stub that raises; only use Numba path when `HAS_NUMBA` and kernel is available. |
| **Comparing Numba vs Python** | Need to run the same sweep with and without Numba. | Use `USE_NUMBA=0` (env or in-code) for the “Python” leg; restore default after. |
| **Relative import** | `from .numba_solve import ...` fails when running as a script. | Try relative import first, then `from numba_solve import ...`. |

When adding Numba to another hot path:

1. Identify the **tight numeric loop** and the **per-item work** that can be pure scalars/small arrays.
2. Extract that into a function that uses only **supported types** (no Python objects, no exceptions inside).
3. Pass in **precomputed constants** as scalars or simple arrays; keep the signature simple.
4. Use **`math`** for scalar math in the kernel.
5. Add a **warm-up** call before any timed run so benchmarks reflect steady-state performance.
6. Provide a **fallback** (e.g. `HAS_NUMBA` + Python path) and an **optional disable** (e.g. `USE_NUMBA`) for comparison and environments where Numba is not installed.
