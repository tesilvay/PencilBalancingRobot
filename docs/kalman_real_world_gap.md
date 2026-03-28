# Kalman estimator: simulation vs hardware

This note records why the linear Kalman filter (`KalmanEstimator` in `perception/estimator.py`) behaves well in simulation but poorly on real DVS hardware, what evidence we collected, and what we tried. It is meant for future debugging and onboarding—not as a final design spec.

## Problem statement

- **Simulation:** The Kalman filter tracks well enough that the LQR controller can balance the pencil; benchmarks and single runs look strong.
- **Hardware (real cameras, mock or real actuators):** The same estimator and controller tend to **diverge** (commands saturate, workspace edge). A **low-pass finite-difference** estimator (`LowPassFiniteDifferenceEstimator`) often **feels** more usable because it still responds when vision moves, even if it is noisier on paper.

The gap is **not** explained by a single obvious algebra bug in the Kalman update (predict / innovation / Joseph-style covariance is standard). The main issues are **mismatch between what the filter assumes and what the real pipeline provides**.

## How the filter is wired (brief)

- Built in `core/system_builder.py` with `BuildLinearModel` discretized at **fixed `dt = 0.001` s** (`KalmanEstimator(A, B, dt=0.001, Q, R)`).
- **Important:** `KalmanEstimator.update(..., dt, ...)` receives `dt` from `Perception.update` but **does not use** `dt` in the update; discretization is fixed at construction time. The loop must match that nominal step (see `params.run.dt` and `System` in `simulation/system.py`).
- **Measurements** \(z = [X, \alpha_x, Y, \alpha_y]^T\): positions and tilts only. **Velocities are unobserved**; they come from the prediction model and how innovations map through the gain \(K\).
- **Control in prediction:** \(u = [x_{\mathrm{des}}, y_{\mathrm{des}}]^T\) from `TableCommand` (same as controller output).

## Clues and root causes (ranked)

### 1. Measurement timing: stale `z` at high loop rate (strong)

In **simulation** (`SimVisionModel` in `perception/vision.py`), each step gets a **new** projection of `state_true` plus noise—so at 1 kHz there is a **fresh** \(z\) every millisecond (subject to delay buffer settings).

On **real cameras** (`RealEventCameraInterface.get_observation`), `state_true` is **ignored**. Observations are whatever the background reader threads last stored (`_latest1`, `_latest2`). The main loop may still call `perception.update` at **1 kHz**, but **the same line/pose can be reused for many consecutive steps** until the tracker updates.

**Effect:** The Kalman update assumes each step’s \(z\) is a **new** measurement consistent with the current time. Reusing \(z\) while \(x_{\mathrm{pred}} = A \hat{x} + B u\) advances every step forces the filter to explain a **moving prediction** with a **frozen measurement**. That systematically corrupts **velocity** (and coupled states), which the LQR uses heavily—hence **huge \(\hat{x}_\mathrm{dot}, \hat{y}_\mathrm{dot}\)** (e.g. on the order of **1 m/s**) even when the pencil is nominally still.

**Innovation signature:** Real logs showed \(y = z - H x_{\mathrm{pred}}\) with **long, smooth drifts** in some components (consistent with frozen \(z\)); simulation showed **more step-to-step variation** (consistent with independent noise each step).

### 2. Process vs experiment: null controller and “static hold”

With **`NullController`**, \(u = 0\). The prediction still uses the **linearized balancer / inverted-pendulum-style** `A` matrix. That model is **not** “pencil is rigidly held fixed in the lab frame.” For diagnostics, a **static hold** with \(u = 0\) therefore **does not** isolate the Kalman: the **internal model** keeps predicting dynamics that conflict with “everything is fixed,” so \(\hat{v}\) and \(\hat{\alpha}\) need not settle to zero even when the user intends a static test.

**Separate failure mode:** Simulation at **90°** tilt with null control is **outside** the small-angle linearization; blow-up there reflects **invalid model region**, not necessarily the real-camera bug.

### 3. \(Q\) and \(R\) vs real noise (secondary until timing is fixed)

- `Q = 10^{-6} I\) (very small) → strong trust in the **linear** process model between measurements.
- `R \propto \texttt{variant.noise\_std}^2 I` → tuned to **simulated analytic line noise**, not necessarily to **Hough + regression + calibration** error on hardware (bias, outliers, correlation in time).

Wrong \(R\) or \(Q\) can hurt, but **orders-of-magnitude velocity errors** with **fixed** \(z\) point to **measurement process / timing** first.

### 4. When `get_observation` returns `None`

`Perception.update` substitutes the last estimate or `state_true` (see `perception/vision.py`). On hardware, flicker between valid and invalid line fits can add transients.

### 5. Occasional large \( \hat{x} \) vs \(z\) mismatch in \(x\) (investigate wiring)

In one static log, **vision** \(x \approx +4\) mm was stable while **\(\hat{x} \approx -11\)** mm persisted. That is **not** typical “smooth Kalman lag” if \(R\) is small and many updates have passed. It warrants verifying that the printed **`pose` is exactly the same `PoseMeasurement` passed into `KalmanEstimator.update`**, and that **units and frames** are consistent everywhere (meters vs millimeters in display only, not mixed in the filter).

## What we tested

| Test | What we learned |
|------|------------------|
| Compare sim vs real **innovations** \(y\) | Similar magnitude possible, but **real** sequence looked **smooth / correlated** vs **sim** more **white** step-to-step—consistent with **stale \(z\)**. |
| Log **\(\hat{v}\)** (mm/s) sim vs real | Real showed **~1000 mm/s** scale linear velocities vs sim **tens of mm/s** for comparable-looking pose—controller-breaking. |
| **Static hold** (vision stable) | Confirmed **vision** could be nearly constant while **\(\hat{v}\)** stayed huge—fusion/timing issue, not “vision drift.” |
| **Null controller**, upright pencil, noisy `pose` | **`pose`** jumped few mm / ~1° line-to-line; **`x_hat`** smoothed and did not “lock” to each sample—expected under **noisy \(z\)**. Velocities tens of mm/s—partly **noisy measurements** and partly **\(u=0\)** vs **unstable / mismatched process model** for a held pencil. |
| **Null controller**, sim at **90°** | **Invalid linearization**; estimator/controller explosion—not the primary real-camera diagnosis. |

## Recommended directions (for implementation work)

1. **Align Kalman measurement updates with real vision rate**  
   Run a **full update** only when \(z\) is **new** (e.g. new tracker output or timestamp), and **predict-only** (or hold covariance logic appropriately) between updates—or run the estimator at the camera/tracker rate instead of 1 kHz with duplicated \(z\).

2. **Log for one session**  
   - Raw \(z\) right before `estimator.update` (SI units).  
   - `z_changed` or time since last new line.  
   - Optional: normalized innovations vs \(S = H P H^T + R\) for consistency checks.

3. **Retune \(R\)** (and if needed \(Q\)**)** using **hardware** innovation statistics after timing is honest—not only `noise_std` from sim analytic lines.

4. **Static diagnostic** with **realistic \(u\)** (or fixed table pose matching the experiment), not null control, if the goal is to validate closed-loop-relevant estimation.

## Related code

- `perception/estimator.py` — `KalmanEstimator`
- `perception/vision.py` — `Perception.update`, `SimVisionModel`, `RealEventCameraInterface`
- `core/system_builder.py` — `build_estimator`, `build_vision`
- `simulation/system.py` — `System.step` passes `dt` and command into perception

---

*Last updated from internal debugging notes (Kalman sim vs hardware, innovation and static tests).*
