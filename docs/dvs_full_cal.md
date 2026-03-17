# DVS full calibration (future algorithm)

This document describes a **data-driven, multivariate regression calibration** for the DVS pencil balancer. It is intended as the target design for a future calibration pipeline that replaces the current **intercept-only** 2D lookup with a single learned mapping from all four line observables to full pose. The current calibration is documented in [DVS_CALIBRATION.md](DVS_CALIBRATION.md).

Terminology note (to avoid confusion):

- **camnorm** (camera-normalized): the geometric quantities \((b, s)\) produced by the vision pipeline after converting pixel line fits using camera intrinsics. These are *dimensionless* and are what the analytic `reconstruct()` math expects.
- **standardized** (ML feature scaling): \((x - \mu)/\sigma\) applied to camnorm inputs *only for regression numerical conditioning*. These values must never be fed into the analytic `reconstruct()` directly.

---

## Motivation

- **Current calibration** maps only **(b1, b2) → (X, Y)** and uses the analytic formulas for **(αx, αy)**. It was collected with an **upright pencil**, so it does not excite slopes **(s1, s2)**. When the pencil tilts, (b1, b2) change and we may extrapolate or use a model that does not match the real camera/geometry.
- **Goal:** Let the estimator depend on **all four measured quantities** (b1, s1, b2, s2). Calibration must therefore vary **position and slope** independently. A dataset that only has upright poses excites only (b1, b2); that is why intercept-only calibration is insufficient when tilt matters.
- **Approach:** Treat calibration as **empirical pose estimator identification**: collect (X, Y, αx, αy) ground truth and corresponding (b1, s1, b2, s2) observations over a grid of positions and a small set of tilts, then fit a **regression model** f(b1, s1, b2, s2) → (X, Y, αx, αy). Data-driven calibration tends to beat fragile analytic equations when geometry is slightly wrong and the system is sensitive.

In robotics this style is often called **empirical camera-to-robot calibration**, **system identification with regression**, or (for hand-eye) **hand-eye calibration via regression**. For this document we use:

**Name:** **Multivariate linear regression calibration** / **Empirical pose estimator identification**.

---

## What the estimator depends on

The vision pipeline produces four scalars from the two cameras (all in **camnorm** coordinates):

- **b1, s1** — camnorm intercept and slope from camera 1.
- **b2, s2** — camnorm intercept and slope from camera 2.

The pose is **(X, Y, αx, αy)** (base position in workspace, tilt angles). The calibration problem is to learn a mapping:

```text
f(b1, s1, b2, s2) → (X, Y, αx, αy)
```

So the dataset must **excite all four inputs** (b1, s1, b2, s2). That requires varying both **table position (X, Y)** and **pencil tilt (αx, αy)** in a controlled way.

---

## Dataset design

### Structure: grid × tilts

| X     | Y     | αx   | αy   |
| ----- | ----- | ---- | ---- |
| grid  | grid  | 0    | 0    |
| grid  | grid  | +θ   | 0    |
| grid  | grid  | −θ   | 0    |
| grid  | grid  | 0    | +θ   |
| grid  | grid  | 0    | −θ   |

- **Grid:** e.g. 5×5 positions inside the workspace circle (same idea as current calibration).
- **Tilts:** one upright (0, 0) plus four small tilts: ±θ in αx only and ±θ in αy only.

**Example:** 5 tilts × 25 positions = **125 poses**.

### Tilt magnitude

- Use **about 8–12°**, not large angles (e.g. 45°). The controller will not see 45° in normal operation, and small tilts keep the regression in a well-approximated region and avoid nonlinearities.

### Multiple frames per sample

- Event cameras give noisy slope estimates. For **each** pose (X, Y, αx, αy):
  - Move table to (X, Y) and set tilt (αx, αy) (e.g. with a jig or known fixture).
  - Collect **200–500** camera measurements (b1, s1, b2, s2).
  - Store **averaged** (b1, s1, b2, s2) with ground-truth (X, Y, αx, αy).

So each calibration sample is one row: averaged observables + true pose.

---

## Dataset format

Each calibration sample (one row) should store:

```text
b1
s1
b2
s2
X_true
Y_true
αx_true
αy_true
```

You will have on the order of **125 rows** (for 5×5 grid × 5 tilts).

Recommended storage:

- **Dataset file** (raw samples): save into `perception/calibration_files/` with a name like `dvs_pose_dataset.json` (or `.csv`).
- **Trained model file** (runtime artifact): save into `perception/calibration_files/` with a name like `dvs_pose_regression_model.json`.

Keep these separate: the dataset is for offline training and debugging; the trained model is what you load at runtime.

---

## Regression model

### Target mapping

```text
f(b1, s1, b2, s2) → (X, Y, αx, αy)
```

### Feature vector (linear + interaction terms)

The analytic reconstruct is rational in (b1, b2, s1, s2). In a small operating region, that surface is well approximated by a **low-order polynomial**. Using linear terms plus **interactions** between intercepts and slopes gives the model enough freedom to capture the coupling:

```text
feature_vector = [
    1,
    b1,
    b2,
    s1,
    s2,
    b1*s1,
    b1*s2,
    b2*s1,
    b2*s2
]
```

So **9 features** per sample (one constant, four linear, four interaction).

### Normalization (important)

Standardize camnorm inputs **before** regression to improve numerical stability:

```text
b1' = (b1 - mean(b1)) / std(b1)
```

and similarly for s1, b2, s2.

Important: this is **ML feature scaling** (standardization), not the camnorm pixel→camera conversion. The camnorm conversion still happens upstream in the vision pipeline. The regression takes camnorm values and then standardizes them internally using the **training** mean and std at runtime when applying the model.

---

## Dataset creation

Pseudocode:

```text
dataset = np.array(N_poses,8)
# columns: [b1, s1, b2, s2, X_true, Y_true, ax_true, ay_true]

for pose in calibration_poses:

    move_table_to(pose.X, pose.Y)
    set_tilt(pose.ax, pose.ay)

    measurements = collect_camera_measurements()   # 200–500 frames

    b1 = average(measurements.b1)
    s1 = average(measurements.s1)
    b2 = average(measurements.b2)
    s2 = average(measurements.s2)

    dataset.append([b1, s1, b2, s2, pose.X, pose.Y, pose.ax, pose.ay])
```

### Dataset storage format

We store the calibration dataset as a **NumPy `.npz`** plus a small JSON metadata file:

- `perception/calibration_files/dvs_pose_dataset.npz`
  - key `data`: NumPy array of shape `(N_poses, 8)`
  - **column order**: `[b1, s1, b2, s2, X_true, Y_true, ax_true, ay_true]`
- `perception/calibration_files/dvs_pose_dataset_meta.json`
  - `column_names`: `["b1", "s1", "b2", "s2", "X_true", "Y_true", "ax_true", "ay_true"]`
  - `grid_shape`: e.g. `[5, 5]` for a 5×5 grid
  - `tilt_angles_deg`: e.g. `[0, +θ, -θ]` description for αx/αy tilts
  - optional fields: date, run ID, number of frames averaged per pose, etc.

This way, the `.npz` file is optimized for fast numeric work, and the JSON sidecar documents exactly what each column and pose index means.


## Training (offline)
```
# Compute training mean/std for (b1, s1, b2, s2)
# Standardize (b1, s1, b2, s2) using those stats
# Build design matrix M: each row = feature_vector for one sample

for each sample in dataset:
    b1, s1, b2, s2 = sample.measurements
    feature_vector = [1, b1, b2, s1, s2, b1*s1, b1*s2, b2*s1, b2*s2]
    add feature_vector to matrix M

# Four least-squares problems (one per output)
coeff_X  = least_squares(M, X_true)
coeff_Y  = least_squares(M, Y_true)
coeff_ax = least_squares(M, ax_true)
coeff_ay = least_squares(M, ay_true)
```

We save the training values in a json file in calibration with the following format
```
{
  "model_type": "linear_regression",
  "input_order": ["b1", "s1", "b2", "s2"],
  "input_mean": [0.01, -0.02, 0.00, 0.03],
  "input_std":  [0.10,  0.05, 0.11, 0.04],

  "feature_order": [
    "1",
    "b1_z", "b2_z", "s1_z", "s2_z",
    "b1_z*s1_z", "b1_z*s2_z", "b2_z*s1_z", "b2_z*s2_z"
  ],

  "output_order": ["X", "Y", "ax", "ay"],
  "W": [
    [ /* 9 numbers for X */ ],
    [ /* 9 numbers for Y */ ],
    [ /* 9 numbers for ax */],
    [ /* 9 numbers for ay */]
  ],

  "units": { "X": "m", "Y": "m", "ax": "rad", "ay": "rad" },
  "dataset": { "grid_shape": [5,5], "tilts": "0, ±θ in ax, ±θ in ay" }
}
```

### What to save after training (model artifact)

To run the estimator later you need to serialize:

- **`input_mean`**: `[mean_b1, mean_s1, mean_b2, mean_s2]`
- **`input_std`**: `[std_b1, std_s1, std_b2, std_s2]` (guard against zeros)
- **`feature_definition`**: a string or version (so future code changes don’t silently misinterpret coefficients)
- **`coefficients`**: either four vectors (one per output) or a single matrix `W` of shape `(4, n_features)` for `[X, Y, ax, ay]`

Optionally include metadata: date, grid size, tilt angle, sample count, etc.

---

## Runtime estimator

During operation (real time):

```text
measure camnorm b1, s1, b2, s2

# Standardize with training mean/std (ML feature scaling)
b1', s1', b2', s2' = standardize(b1, s1, b2, s2)

features = [1, b1', b2', s1', s2', b1'*s1', b1'*s2', b2'*s1', b2'*s2']

X  = dot(coeff_X,  features)
Y  = dot(coeff_Y,  features)
ax = dot(coeff_ax, features)
ay = dot(coeff_ay, features)
```

This is a few dot products and can run in **microseconds**. This runtime estimator needs to become a class with the values from the JSON training as its attributes and a single method for the estimation.

Implementation note: it’s best to encapsulate this in one runtime class with an explicit API, so nobody accidentally mixes camnorm and standardized values.

## Naming proposals (to reduce confusion)

### Files

- **Dataset (raw samples)**: `perception/calibration_files/dvs_pose_dataset.json`
- **Trained model (runtime artifact)**: `perception/calibration_files/dvs_pose_regression_model.json`

### Classes

Current position-only calibrator is a lookup/interpolator, so keep “Calibration” there. For the new learned mapping, prefer “Regressor/Model/Estimator” in the name:

- `DVSPoseRegressor`
- `DVSPoseRegressionModel`
- `DVSPoseEstimator`

If it lives in `perception/calibration.py`, consider renaming the module later to `perception/dvs_pose_model.py` (or similar) so “calibration” doesn’t imply “(b1,b2) grid only”.

### Methods

- `load(path)` (class method): load weights + scaler stats
- `predict_pose(b1, s1, b2, s2)` or `estimate(b1, s1, b2, s2)`: returns `(X, Y, ax, ay)`
- `standardize_inputs(b1, s1, b2, s2)`: internal helper (explicitly *not* called “normalize”)

### Variables

- Use `camnorm_*` prefixes for geometric values: `camnorm_b1`, `camnorm_s1`, ...
- Use `std_*` or `_z` for standardized values: `b1_z`, `s1_z`, ...

---

## Why this works

- The true mapping from (b1, s1, b2, s2) to pose is rational (e.g. denominators like 1 + b1*b2). In the **small region of operation**, that surface is very well approximated by a **low-order polynomial**.
- The regression **learns** that polynomial from data, so it automatically adapts to real camera placement, lens, and mounting errors.
- By including **position and tilt** in the calibration, we excite (b1, s1, b2, s2) properly, so the learned map is valid for both upright and slightly tilted pencils.

Conceptually, this is a **learned stereo reconstruction model**: the original paper gave an analytic mapping; here we **learn the mapping empirically**, which is a standard modern approach when the analytic model is imperfect.

---

## Sanity check after calibration

After training:

1. Move the table to **random positions** and tilt the pencil **randomly** within about ±10°.
2. For each pose, compare **predicted (X, Y, αx, αy)** from the regression to **true** pose (e.g. from table command + jig).
3. If **average base error (X, Y) is &lt; 2 mm**, the balancer has a good chance to stabilize. Larger errors may require more data, better normalization, or a slightly richer feature set.

---

## Summary

| Item | Description |
|------|-------------|
| **Name** | Multivariate linear regression calibration / Empirical pose estimator identification |
| **Inputs** | camnorm b1, s1, b2, s2 (then standardized for regression; then optional interaction terms) |
| **Outputs** | X, Y, αx, αy |
| **Dataset** | Grid × 5 tilts (0, ±θ in αx, ±θ in αy); e.g. 5×5 × 5 = 125 poses |
| **Per pose** | 200–500 frames averaged to one (b1, s1, b2, s2) + (X, Y, αx, αy) |
| **Features** | [1, b1', b2', s1', s2', b1'*s1', b1'*s2', b2'*s1', b2'*s2'] where \(b',s'\) are standardized |
| **Training** | Four least-squares fits: same M, targets X_true, Y_true, ax_true, ay_true |
| **Runtime** | Standardize (ML) → build feature vector → four dot products with stored coefficients |

This document is a **spec for a future implementation**. The current system uses the intercept-only 2D calibration described in [DVS_CALIBRATION.md](DVS_CALIBRATION.md).
