# DVS full calibration (future algorithm)

This document describes a **data-driven, multivariate regression calibration** for the DVS pencil balancer. It is intended as the target design for a future calibration pipeline that replaces the current **intercept-only** 2D lookup with a single learned mapping from all four line observables to full pose. The current calibration is documented in [DVS_CALIBRATION.md](DVS_CALIBRATION.md).

---

## Motivation

- **Current calibration** maps only **(b1, b2) → (X, Y)** and uses the analytic formulas for **(αx, αy)**. It was collected with an **upright pencil**, so it does not excite slopes **(s1, s2)**. When the pencil tilts, (b1, b2) change and we may extrapolate or use a model that does not match the real camera/geometry.
- **Goal:** Let the estimator depend on **all four measured quantities** (b1, s1, b2, s2). Calibration must therefore vary **position and slope** independently. A dataset that only has upright poses excites only (b1, b2); that is why intercept-only calibration is insufficient when tilt matters.
- **Approach:** Treat calibration as **empirical pose estimator identification**: collect (X, Y, αx, αy) ground truth and corresponding (b1, s1, b2, s2) observations over a grid of positions and a small set of tilts, then fit a **regression model** f(b1, s1, b2, s2) → (X, Y, αx, αy). Data-driven calibration tends to beat fragile analytic equations when geometry is slightly wrong and the system is sensitive.

In robotics this style is often called **empirical camera-to-robot calibration**, **system identification with regression**, or (for hand-eye) **hand-eye calibration via regression**. For this document we use:

**Name:** **Multivariate linear regression calibration** / **Empirical pose estimator identification**.

---

## What the estimator depends on

The vision pipeline produces four scalars from the two cameras:

- **b1, s1** — normalized intercept and slope from camera 1 (x-axis).
- **b2, s2** — normalized intercept and slope from camera 2 (y-axis).

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

You will have on the order of **125 rows** (for 5×5 grid × 5 tilts). Storage can be CSV, JSON array of objects, or similar; the regression step builds a design matrix from the first four columns and four target vectors from the last four.

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

Normalize inputs **before** regression to improve numerical stability:

```text
b1' = (b1 - mean(b1)) / std(b1)
```

and similarly for s1, b2, s2. Use the **training** mean and std at runtime when applying the model.

---

## Training (offline)

Pseudocode:

```text
dataset = []

for pose in calibration_poses:

    move_table_to(pose.X, pose.Y)
    set_tilt(pose.ax, pose.ay)

    measurements = collect_camera_measurements()   # 200–500 frames

    b1 = average(measurements.b1)
    s1 = average(measurements.s1)
    b2 = average(measurements.b2)
    s2 = average(measurements.s2)

    dataset.append([b1, s1, b2, s2, pose.X, pose.Y, pose.ax, pose.ay])

# Normalize (b1, s1, b2, s2) using training mean/std
# Build design matrix M: each row = feature_vector for one sample

for each sample in dataset:
    b1, s1, b2, s2 = sample.measurements
    feature_vector = [1, b1, b2, s1, s2, b1*s1, b1*s2, b2*s1, b2*s2]
    add feature_vector to matrix M

# Four least-squares problems
coeff_X  = least_squares(M, X_true)
coeff_Y  = least_squares(M, Y_true)
coeff_ax = least_squares(M, ax_true)
coeff_ay = least_squares(M, ay_true)
```

So we solve **four** linear least-squares problems (one per output). The same design matrix M is used for all four; only the target vector changes.

---

## Runtime estimator

During operation:

```text
measure b1, s1, b2, s2

# Normalize with training mean/std
b1', s1', b2', s2' = normalize(b1, s1, b2, s2)

features = [1, b1', b2', s1', s2', b1'*s1', b1'*s2', b2'*s1', b2'*s2']

X  = dot(coeff_X,  features)
Y  = dot(coeff_Y,  features)
ax = dot(coeff_ax, features)
ay = dot(coeff_ay, features)
```

This is a few dot products and can run in **microseconds**.

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
| **Inputs** | b1, s1, b2, s2 (normalized, then optional interaction terms) |
| **Outputs** | X, Y, αx, αy |
| **Dataset** | Grid × 5 tilts (0, ±θ in αx, ±θ in αy); e.g. 5×5 × 5 = 125 poses |
| **Per pose** | 200–500 frames averaged to one (b1, s1, b2, s2) + (X, Y, αx, αy) |
| **Features** | [1, b1, b2, s1, s2, b1*s1, b1*s2, b2*s1, b2*s2]; normalize b1, s1, b2, s2 first |
| **Training** | Four least-squares fits: same M, targets X_true, Y_true, ax_true, ay_true |
| **Runtime** | Normalize → build feature vector → four dot products with stored coefficients |

This document is a **spec for a future implementation**. The current system uses the intercept-only 2D calibration described in [DVS_CALIBRATION.md](DVS_CALIBRATION.md).
