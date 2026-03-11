# Development Guide

How to set up, run, and work with the pencil balancing robot codebase.

---

## Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `scipy`, `matplotlib`, `control`, `opencv-python`, `shapely`, `pynput`, `dv-processing` (for real DAVIS346 cameras).

---

## Running the Simulation

### Modes

| Mode | Command | Description |
|------|---------|-------------|
| Single trial | `python main.py --mode single` | One run, lasts as long as total_time sets it in main.py, real-time pacing, optional 3D viz |total_time applies for all simulations, so set it lower when doing benchmarks so it finishes faster
| Benchmark single | `python main.py --mode benchmark_single` | 200 Monte Carlo trials, progress bar |
| Benchmark all configs | `python main.py --mode benchmark_all_configs` | Sweeps controller/estimator/noise/delay, saves JSON |
| Sweep workspace | `python main.py --mode sweep_workspace` | Varies workspace radius, plots stability |
| Graph results | `python main.py --mode graph_results` | Loads latest benchmark JSON, generates plots |

### Examples

```bash
# One trial (opens visualization)
python main.py --mode single

# Monte Carlo benchmark (200 trials, no GUI)
python main.py --mode benchmark_single

# Full benchmark sweep (saves to results/)
python main.py --mode benchmark_all_configs

# Plot latest benchmark
python main.py --mode graph_results
```

---

## Enabling DVS (Event Camera) Path

By default, vision uses `SimVisionModel` (direct projection).

**Simulated DVS** (no hardware): Set `dvs_cam=True`, leave ports `None`.

**Real DAVIS346** (hardware-in-the-loop): Set `dvs_cam=True` and both ports (serials from `dv-list-devices`):

```python
hardware=HardwareParams(
    servo=True,
    servo_port=None,
    dvs_cam=True,
    dvs_cam_x_port="00000499",   # from dv-list-devices
    dvs_cam_y_port="00000500",
),
```

Run `python -m benchmarks.visualize_dvs_cams` (auto-discovers) or `--cam1 SERIAL1 --cam2 SERIAL2` first to verify cams and Hough line.

---

## Real DVS with Simulated Servos

Use real DVS cameras to measure the plant (physical pencil) while the controller computes commands, but **do not** send commands to physical servos. Shows real DVS footage and a workspace plot (x_des, y_des point + circle).

In `main.py`, set:

```python
hardware=HardwareParams(
    servo=False,           # no physical servos
    dvs_cam=True,
    dvs_cam_x_port="00000499",  # or None for auto-discovery
    dvs_cam_y_port="00000500",
    dvs_algo="sam",       # "hough" or "sam" (Sam uses noise filter)
),
run=RunParams(realtimerender=True, ...),
```

The visualizer shows Cam 1, Cam 2 (real event footage + line overlay) and the Workspace window to the right. The Workspace plots the desired point (x_des, y_des) and the safe-radius circle; the point is clamped to the circle edge when outside (matching the plant/mechanism limits). Ports can be `None` to auto-discover the first two devices. The simulation runs indefinitely; press `q` in any visualization window to quit.

---

## Running the Hough Benchmark

Standalone benchmark for the DVS line-tracking algorithm (no full sim):

```bash
# Falling pencil: Hough vs true state (no controller)
python -m benchmarks.benchmark_hough --mode falling --n_trials 50

# Decay sweep: compare alpha error at different Hough decay values
python -m benchmarks.benchmark_hough --mode decay_sweep

# Static line (original)
python -m benchmarks.benchmark_hough --mode static
```

---

## Real DVS Camera Visualization (before HIL)

Verify both DAVIS346 cameras and Hough line tracking before running full hardware-in-the-loop:

```bash
# Auto-discover and open first two cameras (no serials needed)
python -m benchmarks.visualize_dvs_cams

# Or specify serials from dv-list-devices
python -m benchmarks.visualize_dvs_cams --cam1 SERIAL1 --cam2 SERIAL2

# Match main.py pipeline (Sam + noise filter 30 ms)
python -m benchmarks.visualize_dvs_cams --mode sam --noise-filter-duration 30
```

Shows accumulated events with line overlay per camera. Press 'q' to quit.

---

## Project Layout

```
main.py                 # CLI entrypoint
system_builder.py       # Composition root (builds plant, vision, controller, etc.)
core/                   # Dynamics, control, types
perception/             # Vision, DVS algorithms, estimators, dvs_camera_reader
simulation/             # Simulator loop, runner (real_mode when DVS connected)
experiments/            # Experiment runners, Monte Carlo
benchmarks/             # Hough benchmark, visualize_dvs_cams
analysis/               # Plotting, workspace analysis
results/                # Benchmark JSON outputs (created on save)
docs/                   # Architecture, tasks, this guide
```

---

## Recent Changes

- **Real DVS + simulated servos**: Real DAVIS346 cameras measure the physical pencil; controller computes (x_des, y_des); servos stay disconnected. Run indefinitely until `q` or pencil fall.
- **DVSWorkspaceVisualizer**: Three windows — Cam 1, Cam 2 (event footage + line overlay), Workspace (circle + desired point). Point clamped to circle edge when outside limits.
- **`dvs_algo`**: Choose `"hough"` or `"sam"` in `HardwareParams`.
- **`dvs_noise_filter_duration_ms`**: `None` = no filter; `30` = default (Sam only). Use `5`–`10` for low-latency, or `None` to disable.
- **`estimator_lpf_alpha`**: `None` = LPF default (0.95); `0.99` for lower phase lag in real-time.
- **Device discovery**: Set both `dvs_cam_x_port` and `dvs_cam_y_port` to `None` to auto-discover cameras.
- **Quit key**: Press `q` in any visualization window to exit indefinite real-DVS runs.

---

## Key Configuration

- **`main.py`**: `ExperimentSetup` — params, camera_params, default_variant
- **`core/sim_types.py`**: `PhysicalParams`, `HardwareParams` (servo, dvs_cam, dvs_cam_x_port, dvs_cam_y_port, dvs_algo, dvs_noise_filter_duration_ms), `RunParams` (estimator_lpf_alpha), `BenchmarkVariant`, `ExperimentSetup`
- **`system_builder.py`**: Vision selection (`dvs_cam`, `dvs_algo`), controller/estimator choice, DVSWorkspaceVisualizer when real DVS + realtime

---

## Tests

No formal test suite yet. Manual checks:

- `python main.py --mode single` — completes without error
- `python main.py --mode benchmark_single` — 200 trials, reasonable stability
- `python -m benchmarks.benchmark_hough --mode falling` — Hough algorithm
- `python -m benchmarks.visualize_dvs_cams` or `--cam1 S1 --cam2 S2` or `--mode sam` — real DAVIS346 (with hardware)
