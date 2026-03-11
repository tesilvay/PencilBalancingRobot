# Task: Hough Tracker Progress

**Status:** Refactor implemented — real-hardware validation still needed  
**Goal:** Document what was wrong with the previous Python Hough tracker, why it lagged on real DVS data, what changed, and what those changes were based on.

---

## Summary

The previous Python `PaperHoughLineAlgorithm` was not actually matching the original `Conradt/PencilBalancer.java` behavior closely enough for real DVS use.

It worked reasonably in the synthetic benchmark, but it showed noticeable lag on real cameras in `main.py`.

The root issue was not just parameter tuning. The algorithm itself had drifted away from the original update rule:

- Python used a fixed once-per-batch decay.
- Python gave every event equal weight.
- Python solved once after the whole batch.
- Java updated recursively per event.
- Java weighted each event by how close it was to the current line estimate.
- Java tied forgetting to that inlier weight, which made it much more robust to clutter and trailing motion events.

So the fix was to restore the original event-driven behavior while rewriting the code into a cleaner Python structure.

---

## What Was Wrong

### 1. The Python tracker was not honoring the original update rule

The earlier Python version maintained quadratic accumulators and solved the same closed-form line fit, but it updated them like this:

- decay all coefficients once per batch
- add every event with full weight
- solve once at the end

That is not what the original Java code does.

### 2. Real DVS data is not the same as the synthetic benchmark

The synthetic benchmark in `benchmarks/benchmark_hough.py` feeds clean points sampled directly from the current true line.

Real DVS data contains:

- packetized event timing
- uneven event density
- background activity
- leading/trailing motion structure
- stale events from recent motion

If every event is accepted equally, the fitted line is pulled toward a recent motion history cloud instead of the current pencil position.

### 3. The old `decay` parameter was hiding the real problem

In the older Python tracker, reducing `decay` made the estimate more responsive, but that was compensating for the wrong update rule rather than reproducing the original algorithm.

This is why a very aggressive value such as `0.7` could seem necessary on real hardware even though the synthetic falling-pencil benchmark looked acceptable with much slower memory.

---

## What The Original Java Actually Did

Reference: `Conradt/PencilBalancer.java`

For each incoming event:

1. Solve the current line estimate from the current quadratic coefficients.
2. Measure the event's horizontal error relative to that line.
3. Convert that error to a Gaussian inlier weight:

\[
w = \exp\left(-\frac{e^2}{2\sigma^2}\right)
\]

4. Compute an adaptive forgetting factor:

\[
dec = (1 - \text{polyMixingFactor}) + \text{polyMixingFactor}(1 - w)
    = 1 - \text{polyMixingFactor} \cdot w
\]

5. Decay the quadratic coefficients by `dec`.
6. Add the weighted event contribution to the quadratic coefficients.

This means:

- inlier events strongly update the tracker
- outlier/background events have very small influence
- forgetting is tied to useful events, not blindly applied once per packet

That behavior is a major reason the original tracker was faster and more robust on live event streams.

---

## What We Changed

### 1. Restored Java-style per-event recursive updates

`perception/dvs_algorithms.py` now updates the Hough state one event at a time, instead of one packet at a time.

The tracker now:

- solves the current centered line before processing each event
- computes the event residual against that line
- computes a Gaussian inlier weight
- applies adaptive forgetting based on that weight
- accumulates the weighted event into the quadratic state

### 2. Added a bootstrap reset that matches the Java behavior

The tracker reset now seeds two synthetic vertical-line points, just like the Java code.

This avoids the tracker starting from a degenerate zero state and gives it a valid initial line estimate immediately.

### 3. Replaced opaque scalar state with readable dataclasses

The refactor groups Hough-related state/config into `core/sim_types.py`:

- `HoughQuadraticState`
- `HoughTrackerParams`

This preserves the math while making the update path easier to read and maintain.

### 4. Replaced fixed `decay` tuning with Java-style Hough parameters

The main Hough tuneables are now:

- `mixing_factor`
- `inlier_stddev_px`
- `min_determinant`

These correspond much more closely to the original algorithm than the old batch-level `decay` knob.

### 5. Threaded the new parameters through app and benchmark entrypoints

The Hough parameter block now lives in `HardwareParams.dvs_hough` and is surfaced in:

- `main.py`
- `system_builder.py`
- `benchmarks/visualize_dvs_cams.py`
- `benchmarks/benchmark_hough.py`

That way the same mental model is used for real DVS, standalone visualization, and synthetic tests.

---

## What This Was Based On

The refactor was based on direct comparison to the original Java implementation in:

- `Conradt/PencilBalancer.java`

Specifically, the Python rewrite now follows the same high-level event update sequence as:

- `updateCurrentEstimateX/Y()`
- `polyAddEventX/Y()`
- `resetPolynomial()`

It does **not** copy the Java naming or structure literally. The goal was:

- keep the original algorithm
- improve readability
- isolate tracker math from unrelated controller logic

---

## What Has Been Verified

Verified so far:

- the Python Hough tracker now uses Java-style recursive event updates
- the new Hough parameters are wired through config and benchmarks
- the synthetic Hough benchmark still runs successfully after the refactor

Not yet verified:

- live DAVIS346 performance after the refactor
- whether `main.py` now has acceptable real-world lag
- final tuned values for `mixing_factor` and `inlier_stddev_px`

So this document records an implementation and reasoning update, not a final claim that the Hough path is production-ready.

---

## Recommended Next Validation

1. Test `python -m benchmarks.visualize_dvs_cams --mode hough` first.
2. Tune `--hough-mixing-factor` and `--hough-inlier-stddev-px` against the real pencil.
3. Then run `main.py` with the same Hough settings and compare perceived lag in:
   - the camera line overlays
   - the workspace target visualization
4. Only after that, decide whether remaining lag is still in the tracker or in downstream timing/synchronization.

---

## Relevant Files

- `Conradt/PencilBalancer.java`
- `perception/dvs_algorithms.py`
- `core/sim_types.py`
- `system_builder.py`
- `main.py`
- `benchmarks/visualize_dvs_cams.py`
- `benchmarks/benchmark_hough.py`
- `docs/TASKS/DVS_FULL_SIM.md`
