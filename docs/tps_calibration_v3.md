# TPS Calibration v3 — Method, Motivation, and Extension Guide

## Background: why v1 and v2 failed at the corners

The DVS sensor pipeline estimates the pencil's world position `(px, py)` and tilt angles `(α_x, α_y)` from two camera observations. Each camera produces a line observation: an intercept `b` (pixel column where the pencil crosses the camera's mask row) and a slope `s` (pixels/pixel).

**v1 (affine, per-axis):** independently fit `px = k1·b1 + c1` and `py = k2·b2 + c2`. Each axis ignores the other camera entirely.

**v2 (bilinear, 6 params):** fit `px = a0 + a1·b1 + a2·b2` and `py = c0 + c1·b2 + c2·b1`. Cross-terms `a2`, `c2` correct for first-order perspective coupling.

Both fail at the workspace corners because:
- The true mapping from world position to pixel intercepts is a **planar homography** (perspective division — rational, not polynomial).
- **Lens distortion** is largest near the image corners, exactly where workspace corners project.
- **Physical asymmetry** (camera pose errors, optical-axis roll) creates cross-coupling that symmetric polynomials cannot absorb.
- **Tilt projection** varies with position: the same physical tilt angle produces a different pixel slope at `(4, 4) cm` than at `(0, 0) cm`. v1/v2 model tilt as a 1D function of angle only, making tilt estimates drift at the corners.

**Symptom:** the model cannot recover its own calibration points. Corner drift at `(±4, ±4) cm` is worst. Adding more samples into a mis-specified model cannot fix this.

---

## v3: Thin-Plate Spline (TPS) non-parametric interpolation

A **thin-plate spline** (TPS) is a radial-basis-function (RBF) interpolant that:
- Passes **exactly** through every calibration sample (exact interpolation, zero residual at training points).
- Is the smoothest possible surface through those points in the bending-energy sense.
- Requires no assumption about the functional form of the underlying mapping.
- Extrapolates reasonably near the convex hull of training data; becomes unreliable far outside it.

We use `scipy.interpolate.RBFInterpolator` with `kernel="thin_plate_spline"`.

### What is calibrated

**Position (2D → 2D):** at each grid point `(px_m, py_m)` in the workspace, record the pixel intercepts `(b1_px, b2_px)` observed on both cameras. Train two RBF maps:

```
rbf_b1 : (px_m, py_m) → b1_px      (world position → cam1 intercept)
rbf_b2 : (px_m, py_m) → b2_px      (world position → cam2 intercept)
```

**Tilt (3D → 1D):** at each of `N_pos` positions and `N_ang` tilt angles, record the pixel slope. Train two RBF maps:

```
rbf_s1 : (px_m, py_m, α_x_rad) → s_px_cam1
rbf_s2 : (px_m, py_m, α_y_rad) → s_px_cam2
```

This makes tilt estimation position-aware: the same slope on cam1 correctly maps to different `α_x` values depending on where in the workspace the pencil currently is.

### Sample grids

**Position grid:** 2 cm step over the 7 cm workspace circle → ~37 points.

```
x, y ∈ {-6, -4, -2, 0, 2, 4, 6} cm,  keep if hypot(x, y) ≤ 7 cm
```

**Tilt grid:** 5-point cross × 5 angles = 25 samples per camera.

```
positions : (0,0), (±4, 0), (0, ±4) cm
angles    : -10°, -5°, 0°, +5°, +10°
```

**More points = better.** The TPS interpolant is exact at training points but smoother / more accurate between them when training density is higher. A 1 cm position grid (≈145 points) and a 9-point tilt cross (plus9 pattern, 45 tilt samples) would significantly improve interpolation quality in the workspace interior.

### Self-recovery property

Because TPS is an exact interpolant, the model always recovers its own calibration points to machine precision. This is a built-in sanity check: if `pixel_to_world(b1_calib, b2_calib)` does not return the corresponding `(px_m, py_m)`, something is wrong with the data, not the model.

---

## Runtime inference

### Forward (world → pixel)

```python
b1_pred = rbf_b1([[px_m, py_m]])[0]   # used for warm-starting the calibrator
b2_pred = rbf_b2([[px_m, py_m]])[0]
```

### Inverse (pixel → world): Newton iteration

Given observed `(b1_obs, b2_obs)`, solve `F(px, py) = 0` where:

```
F = [rbf_b1(p) - b1_obs,
     rbf_b2(p) - b2_obs]
```

Algorithm:
1. **Seed** from the nearest training sample by pixel-space distance.
2. **Jacobian** by forward finite differences (step ~1e-4 m).
3. **Iterate** `p_{k+1} = p_k - J⁻¹ · F(p_k)` up to 10 steps, stop when `‖F‖ < 1e-3 px`.
4. **Clamp** result to workspace circle; emit a warning if the solution hit the boundary.

Typical convergence: 2–4 iterations.

### Inverse (slope → tilt angle): 1D root-find

Given `(px_m, py_m)` and observed slope `s_obs`, solve:

```
rbf_s1(px_m, py_m, α) = s_obs
```

using `scipy.optimize.brentq` over `[α_min, α_max]` (the calibrated range, ±10°).

**Extrapolation beyond the calibrated range:** when `s_obs` is outside the achievable slopes at `[α_min, α_max]`, the model linearly extrapolates using the average `dS/dα` across the calibrated band:

```python
df_da    = (f_hi - f_lo) / (a_hi - a_lo)          # avg slope sensitivity
alpha    = anchor_boundary - f_at_anchor / df_da   # linear extension
alpha    = clip(alpha, -max_tilt_rad, +max_tilt_rad)
```

The default clamp is **±25°**, well beyond the ±10° training range. This is intentional: accuracy degrades outside the training range, but the extrapolated value is useful for fall detection (detecting extreme tilts) even if it is not numerically precise.

---

## Serialization

The JSON file stores only raw samples. RBF interpolators are rebuilt from scratch every time the file is loaded (they are never serialized). This means:

- Adding calibration points only requires re-running the calibrator and overwriting the file.
- The model is always consistent with the stored samples.

JSON structure (version tag `"v3_tps"` distinguishes it on load):

```json
{
  "version": "v3_tps",
  "mask_y_cam1": 160,
  "mask_y_cam2": 150,
  "positions_m": [[-0.06, -0.02], ...],
  "b1_px": [...],
  "b2_px": [...],
  "tilt_positions_m": [[0.0, 0.0], [0.04, 0.0], ...],
  "tilt_alphas_rad": [-0.1745, -0.0873, 0.0, 0.0873, 0.1745],
  "tilt_s_px_cam1": [...],
  "tilt_s_px_cam2": [...],
  "metadata": { ... }
}
```

---

## Applying TPS to the actuator

The actuator has the same fundamental problem: the mapping from a **commanded table position** `(px_cmd, py_cmd)` to the **actual observed table position** `(px_actual, py_actual)` is not a simple linear relationship. Backlash, mechanical flex, servo nonlinearity, and position-dependent coupling all contribute non-parametric distortions.

### What to calibrate

Command a grid of positions and record the actual positions (from the DVS sensor or another reference). Train:

```
rbf_actual_x : (px_cmd, py_cmd) → px_actual
rbf_actual_y : (px_cmd, py_cmd) → py_actual
```

Then the **inverse feedforward correction** solves:

```
find (px_cmd, py_cmd) such that rbf_actual(px_cmd, py_cmd) ≈ (px_desired, py_desired)
```

using the same Newton iteration pattern already implemented for the sensor inverse.

### Torque / angle mapping (if applicable)

If the servo is commanded by angle (not position), the same pattern applies to the tilt channel:

```
rbf_torque : (α_cmd_x, α_cmd_y) → (α_actual_x, α_actual_y)
```

with a 2D grid over the command space.

### Sample strategy for the actuator

The same density argument applies: more grid points → better interpolation. Recommended starting grid:

| Grid step | Points in 7 cm circle | Time @ 5 s settle |
|-----------|------------------------|-------------------|
| 4 cm      | 9                      | ~1 min            |
| 2 cm      | 37                     | ~3 min            |
| 1 cm      | 145                    | ~12 min           |

Automate with the servo: command each point, wait for settle, read DVS output, log `(cmd, actual)`. No operator is needed if the DVS sensor is already working.

### Key implementation notes

- The TPS interpolant for the actuator is a **correction map**, not a replacement for the controller. The controller still outputs a desired position; the TPS converts it to a corrected command before sending to the servo.
- Extrapolation beyond the command grid is dangerous for the actuator (can command hardware out of range). Clamp strictly to the calibrated region, unlike the sensor which allows loose extrapolation for fall detection.
- The actuator RBF can be warm-started from a previous calibration the same way the sensor calibrator warm-starts from the previous model — most points will need only a small nudge.

---

## Files

| File | Role |
|------|------|
| [simple_dvs_regression_model.py](../src/system/sensor/observation_model/simple_dvs_regression_model.py) | `TPSCalibrationV3` dataclass, `SimpleDVSRegressionModel` with v3 dispatch, `save_tps_v3_calibration`, `pixel_to_world`, `slope_to_alpha_x/y` |
| [simple_dvs_regression_calibrator.py](../src/system/sensor/calibration_tool/simple_dvs_regression_calibrator.py) | `TiltGrid_Samples`, `TiltGridCam1/2Stage`, `_v3_sample_arrays`, CLI `--v3` flag and `main()` wiring |
| [simple_dvs.py](../src/system/sensor/observation_model/simple_dvs.py) | `SimpleDVSRegressionModelLoader` — registry-facing wrapper that calls `load()` and prints model info |
| [calibration_files/simple_dvs_regression.json](../src/system/sensor/observation_model/calibration_files/simple_dvs_regression.json) | Active calibration file (v1/v2/v3 depending on what was last saved) |
