# Task: Full DVS Simulation with Hough Space

**Status:** Implemented — testing remaining  
**Goal:** Add Hough space to the full simulation, simulating a full DVS camera pipeline, and support real DAVIS346 hardware.

---

## Current State

### What Exists

1. **Three vision paths** (selected in `system_builder.build_vision` via `params.hardware.dvs_cam` and ports):
   - **`SimVisionModel`** (default): Direct projection of true state → `CameraPair`. Adds noise and optional delay. No events, no DVS.
   - **`SimEventCameraInterface`**: Simulates DVS by generating events from true state, feeding them to a line algorithm, returning `CameraPair`. Uses `PaperHoughLineAlgorithm`.
   - **`RealEventCameraInterface`**: Real DAVIS346 cameras via dv-processing. Background threads read events, Hough runs continuously. `get_observation(None)` returns latest estimate. Used when both `dvs_cam_x_port` and `dvs_cam_y_port` are set.

2. **`SimEventCameraInterface`** (`perception/vision.py`):
   - `generate_events(b, s, n=200)`: Samples points along the line in pixel space, adds noise, filters to in-frame.
   - `get_observation(state_true)`: Projects state → true line → generates events → Hough → `CameraPair`.

3. **`RealEventCameraInterface`** (`perception/vision.py`):
   - Uses `DVSReader` to open two DAVIS346 cameras by serial.
   - Background threads: read events → `algo.update()` → store latest pixel-space `CameraObservation`.
   - `get_observation(state_true)`: Ignores state, returns latest `CameraPair` (pixel→normalized).

4. **`DVSReader`** (`perception/dvs_camera_reader.py`):
   - Wraps dv-processing. Opens camera by serial, `get_event_batch()`, `close()`.

5. **`PaperHoughLineAlgorithm`** (`perception/dvs_algorithms.py`):
   - Quadratic accumulators, decay=0.95 (responsive). Returns `CameraObservation` or `(None, None)`.

6. **Unified runner** (`simulation/simulation_runner.py`):
   - When `dvs_cams_connected(params)`: `real_mode=True`, Simulator skips `plant.step()`, returns `state_est`. Hardware-in-the-loop ready.

7. **Standalone benchmarks**:
   - `benchmarks/benchmark_hough.py`: Static / falling pencil / decay sweep.
   - `benchmarks/visualize_dvs_cams.py`: Init both cams, render events + Hough line overlay. Use before HIL.

### Relevant Files

| File | Role |
|------|------|
| `perception/vision.py` | `SimVisionModel`, `SimEventCameraInterface`, `RealEventCameraInterface` |
| `perception/dvs_camera_reader.py` | `DVSReader` for DAVIS346 |
| `perception/dvs_algorithms.py` | `PaperHoughLineAlgorithm`, `SurfaceRegressionAlgorithm` |
| `simulation/simulator.py` | `real_mode` branch: skip plant when real DVS |
| `simulation/simulation_runner.py` | `dvs_cams_connected` → `real_mode` |
| `benchmarks/visualize_dvs_cams.py` | Verify cams + Hough before HIL |

---

## Remaining: Testing

Most implementation is complete. Remaining work:

1. **Optional**: Run `benchmark_single` with real cams to quantify stability.

---

## Future directions (optional)

1. **Explicit Hough space**: Replace or augment the quadratic accumulators with an explicit Hough parameter space (e.g. θ–ρ bins), per the paper’s "Gaussian in Hough space" description.

2. **More realistic event simulation**: Improve `SimEventCameraInterface.generate_events()`:
   - Temporal structure (event timestamps, rate)
   - More realistic event distribution (e.g. brightness change threshold)
   - Possibly integrate with plant dynamics for motion blur / event density

3. **Benchmark DVS in full sim**: Run `benchmark_single` and `benchmark_all_configs` with `dvs_cam=True` to compare DVS vs direct projection (noise, delay, stability).

4. **Unify Hough benchmark with full sim**: Ensure `benchmark_hough.py` and `SimEventCameraInterface` share the same event-generation logic and that the full sim uses the same DVS pipeline.

---

## How to Enable DVS

**Simulated DVS** (no hardware): In `main.py`, set `dvs_cam=True`, leave ports `None`. Uses `SimEventCameraInterface`.

**Real DAVIS346**: Set `dvs_cam=True`, `dvs_cam_x_port=None`, `dvs_cam_y_port=None`. Uses `RealEventCameraInterface`; runner enters real mode (skips plant). cams automatically grab the serial port

---

## Resolved: Hough Lag

The default `PaperHoughLineAlgorithm` decay=0.999 caused severe lag (~14° mean alpha error) on a falling pencil. The full sim uses **decay=0.95** for responsiveness (~3.8° mean error, 1.1° median). Run `python -m benchmarks.benchmark_hough --mode decay_sweep` to compare.

---

## Latency: Real-Time Hybrid Sim vs Original sam_cam.py

**Symptom:** When running real-time hybrid sim with real DVS cameras (`dvs_algo="sam"`), the pencil/line overlay has noticeable latency compared to the standalone `sam_cam.py` script.

### Pipeline Comparison

| Component | Original sam_cam.py | Project (main.py + SAM) |
|-----------|---------------------|-------------------------|
| Event source | `getNextEventBatch()` | Same, via `DVSReader` |
| Noise filter | `BackgroundActivityNoiseFilter` 30 ms | Same when `dvs_algo="sam"` |
| Line fit | OLS on filtered batch | `SamLineAlgorithm` (equivalent) |
| Downstream | Direct display | Reconstruct → Estimator → Controller → Display |

### Identified Latency Sources

1. **BackgroundActivityNoiseFilter (30 ms)** — Both use it. Events must persist for 30 ms before output. This is the largest single source of latency.

2. **visualize_dvs_cams vs main.py** — `visualize_dvs_cams` does *not* use the noise filter (DVSReader default `use_noise_filter=False`). `main.py` with `dvs_algo="sam"` uses it. Standalone visualization can feel more responsive because it avoids the 30 ms filter.

3. **Estimator phase lag** — `LowPassFiniteDifferenceEstimator` (alpha=0.9) smooths velocities and adds phase lag. The workspace shows the controller command `(x_des, y_des)`, which depends on this estimate.

4. **Event batching** — `getNextEventBatch()` may buffer events; timing depends on dv-processing.

### Recommended Fixes (in order of impact)

1. **Reduce or disable noise filter for low-latency mode**
   - Add `backgroundActivityDuration` as a configurable parameter (e.g. 5–10 ms instead of 30 ms).
   - Or add a "low latency" mode that disables the filter when noise is acceptable.

2. **Align visualize_dvs_cams with main.py**
   - When comparing behavior, run `visualize_dvs_cams` with the same filter settings as `main.py` (e.g. add `--noise-filter` flag).

3. **Tune estimator for real-time**
   - For real-time use, consider `FiniteDifferenceEstimator` (no smoothing) or a less aggressive low-pass (higher alpha, e.g. 0.95–0.99).

4. **Optional: surface-based regression**
   - The original `sam_cam.py` uses a surface (decay=0.5 every 5 frames) for display only; line fit is on raw events. `SurfaceRegressionAlgorithm` fits on surface points (threshold + morphology). For low latency, prefer per-batch OLS (`SamLineAlgorithm`) over surface accumulation.

### References (Latency)

- `sam_cam.py` — original standalone script
- `perception/dvs_algorithms.py` — `SamLineAlgorithm`, `SurfaceRegressionAlgorithm`
- `perception/dvs_camera_reader.py` — `DVSReader`, `use_noise_filter`
- `perception/vision.py` — `RealEventCameraInterface._reader_loop`

---

## References

- Conradt et al., "A Pencil Balancing Robot using a Pair of AER Dynamic Vision Sensors," IEEE, 2009
- `docs/architecture.MD` — Hough / camera-line pipeline, real DVS
