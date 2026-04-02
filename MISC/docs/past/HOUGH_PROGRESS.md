# Task: Hough Tracker Progress

**Status:** Numba JIT + batch draining implemented and verified on live hardware  
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
- on live cameras, a bad Hough lock can happen at initialization if the pencil is not yet the dominant structure in view
- the bad `cam1` behavior observed in one session was traced to that initialization / clutter issue rather than an intentional code difference between camera 1 and camera 2
- **live DAVIS346 performance after the Numba JIT + batch-draining fix: no visible lag in `benchmarks/visualize_dvs_cams.py --mode hough`** (2026-03)

Not yet verified:

- whether `main.py` now has acceptable real-world lag with the same changes
- final tuned values for `mixing_factor` and `inlier_stddev_px`

---

## Live Hardware Findings (2026-03)

### 1. The current cam1 failure mode does not appear to come from different tracker code

The current Python path constructs the two camera trackers symmetrically:

- same `PaperHoughLineAlgorithm` class
- same `HoughTrackerParams`
- same event-to-line update rule

The only practical differences are:

- which physical device is assigned to `cam1` vs `cam2`
- the actual event stream each physical camera produces
- scene geometry, focus, clutter, and event density

In testing, the "cam1 is broken" symptom was traced to the tracker latching onto the wrong object when the pencil was not yet the dominant line-like structure in the scene. Once initialized on the correct object, the failure looked much more like a basin-of-attraction / clutter problem than a cam1-specific code-path bug.

### 2. The remaining major issue is real-time lag under high event rate

The main live symptom now is:

- Hough can look precise and robust once locked
- but it can lag by roughly 1 second under strong motion
- larger motion causes more lag, which strongly suggests packet backlog caused by event-rate spikes

This is consistent with the current Python implementation:

- `PaperHoughLineAlgorithm.update()` loops over events one by one in Python
- for each event it computes residual, Gaussian weight, adaptive forgetting, accumulator update, and a line solve
- if event bursts arrive faster than this loop can process them, the estimate falls behind wall-clock time

By contrast, `SamLineAlgorithm` is much cheaper because it performs a direct OLS fit on the whole batch using vectorized NumPy operations. That likely explains why `sam` can feel real-time while Hough does not, even when both use the same camera source and similar visualization.

### 3. Visualization is probably not the root cause, but display work can still add load

The line overlay itself is not likely the problem:

- both Sam and Hough ultimately return only slope and intercept
- the visualizer only draws one line per camera

However, event-surface accumulation and display still cost CPU:

- building display surfaces from dense event streams is not free
- `np.add.at(...)` on many random pixel indices can become expensive under heavy motion
- if visualization shares the same thread as camera ingest / Hough, that extra work increases backlog

So visualization is probably a secondary contributor, not the primary cause.

### 4. What the original Conradt system did differently

The paper and Java code indicate that the original system relied on an architecture that was better suited to high event rate:

- two independent camera-processing threads
- packets processed as soon as they arrived
- visualization in a lower-priority thread
- strict stereo timestamp ordering intentionally relaxed so packets could be delivered immediately

The paper reports:

- up to about 3 million events/s processing capability
- roughly 200k-300k events/s during balancing
- packet intervals on the order of 125-300 microseconds

That architecture matters as much as the line-tracker math. A mathematically correct Python port can still lag badly if the event update path is much slower than the original compiled Java / jAER environment.

---

## Numba JIT + Batch Draining Fix (2026-03)

### What was done

The ~1 s lag under motion was caused by the pure-Python per-event Hough loop being too slow for live DVS event rates (200k-300k events/s during balancing). Two changes fixed it:

1. **Numba JIT compilation of the Hough inner loop.** The per-event update (line solve, Gaussian weight, adaptive forgetting, accumulation) was extracted into `_hough_update_events_jit()`, a standalone `@numba.njit(cache=True)` function in `perception/dvs_algorithms.py`. It takes flat float64 arrays and scalar state, returns updated state. No Python objects cross the JIT boundary. The first call compiles (~1-2 s), then the result is cached to disk. Subsequent calls run at native speed.

2. **Batch draining in `benchmarks/visualize_dvs_cams.py`.** The main loop now drains all queued batches per camera per iteration (reads until `get_event_batch()` returns `None`), concatenates them, and processes the merged batch. This prevents backlog accumulation when processing takes longer than inter-packet intervals. When no events are available from either camera, the loop sleeps briefly to avoid busy-waiting.

3. **Optional event cap (`max_events`).** `PaperHoughLineAlgorithm` now accepts a `max_events` constructor parameter. When set, `update()` keeps only the newest N events (tail slice). Exposed via `--max-events-per-batch` in the benchmark CLI. Default is `None` (no cap), since Numba handles typical rates.

### Why this matches the original Java approach

The original `Conradt/PencilBalancer.java` solved the lag problem by running in compiled JVM code fast enough to process every event. The Numba JIT achieves the same thing: make the per-event loop fast rather than skip events.

### Development note

The DAVIS346 cameras require USB access permissions. The only way to run the benchmark during development was:

```
sudo .venv/bin/python -m benchmarks.visualize_dvs_cams --mode hough
```

### Result

After these changes, `benchmarks/visualize_dvs_cams.py --mode hough` runs with no visible lag on live DAVIS346 cameras, including under fast pencil motion.

### Files changed

- `perception/dvs_algorithms.py` — added `_hough_update_events_jit()`; refactored `PaperHoughLineAlgorithm.update()` to call it; added `max_events` parameter
- `benchmarks/visualize_dvs_cams.py` — batch-draining main loop; `--max-events-per-batch` CLI option
- `requirements.txt` — added `numba>=0.58`

---

## Possible Solutions / Next Changes

### A. Improve initialization and reduce wrong-object lock

Possible changes:

- require an explicit initialization step with the pencil already centered in each camera
- add a manual reset / re-lock hotkey in the benchmark viewer
- restrict Hough updates to a region of interest around the expected pencil location
- add a lightweight prefilter to suppress obvious background clutter before Hough
- allow per-camera Hough parameters instead of forcing both cameras to share the same values

Why:

- the current tracker can lock onto a strong non-pencil edge if that is what dominates early events
- once the Gaussian weighting is centered on the wrong structure, it may keep reinforcing that wrong structure

### B. Reduce backlog by changing how events are processed when rate spikes

Possible changes:

- cap the number of events processed per batch
- process only the newest part of a large packet instead of every event in the packet
- skip stale packets when fresher packets are already waiting
- subsample events during overload
- process small chunks of events instead of strict event-by-event recursion, if that approximation is acceptable

Why:

- the current pure-Python per-event loop likely becomes the dominant bottleneck during large motion bursts
- if the system is behind real time, it is usually better to lose some event detail than to keep processing old data 1 second late

### C. Decouple ingest, tracking, and visualization more aggressively

Possible changes:

- run one reader / tracker thread per camera in the benchmark path, not just in the full `RealEventCameraInterface`
- keep the latest line estimate and latest display surface per camera
- refresh the GUI at fixed 24-30 FPS by sampling the most recent state, rather than tying display to packet processing
- keep rendering lower priority than event ingestion and tracking

Why:

- this matches the original Conradt architecture more closely
- it prevents display work from directly slowing event processing

### D. Profile before making large algorithmic changes

Recommended measurements:

- event count per packet
- time spent in `get_event_batch()`
- time spent in display-surface accumulation
- time spent in `PaperHoughLineAlgorithm.update()`
- time spent in rendering

Useful comparisons:

- Sam vs Hough on the same live scene
- one camera vs two cameras
- low motion vs high motion
- display on vs display off

Why:

- this will confirm whether the main bottleneck is really the Hough loop, display accumulation, packet delivery, or some combination

### E. Consider moving the Hough hot path out of pure Python

Possible changes:

- Numba
- Cython
- a small C/C++ extension
- a more vectorized approximation of the recursive update

Why:

- the current implementation is faithful to the Java logic, but Python is a poor fit for a heavy per-event recursive inner loop at live DVS rates
- if the recursive event update is kept exactly as-is mathematically, compiling that hot loop may be the cleanest route to real-time performance

---

## Recommended Next Validation

~~1. Test `python -m benchmarks.visualize_dvs_cams --mode hough` first.~~ **Done (2026-03).** No visible lag after Numba JIT + batch draining.

2. Verify camera identity explicitly using serials instead of relying on discovery order.
3. Use the raw DVS calibration preview to confirm that each camera sees a clean pencil event stream before enabling Hough.
4. Tune `--hough-mixing-factor` and `--hough-inlier-stddev-px` against the real pencil.
5. Profile Sam vs Hough with timing around packet read, surface accumulation, Hough update, and rendering.
6. Then run `main.py` with the same Hough settings and compare perceived lag in:
   - the camera line overlays
   - the workspace target visualization
7. Only after that, decide whether remaining lag is still in the tracker or in downstream timing/synchronization.

---

## Relevant Files

- `Conradt/PencilBalancer.java`
- `perception/dvs_algorithms.py`
- `core/sim_types.py`
- `system_builder.py`
- `main.py`
- `benchmarks/visualize_dvs_cams.py`
- `benchmarks/benchmark_hough.py`
- `hardware/dvs_camera_calibration.py`
- `docs/TASKS/DVS_FULL_SIM.md`
