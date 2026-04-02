# Hough Space Demo and Math

This folder adds an interactive explanation of the Hough-space line tracker used in this repo.

Run the demo with:

```bash
python explanations/hough_space_demo.py
```

The window has four panels:

- `Point Space`: click to add points in the original `(x, y)` plane.
- `Each Point Becomes a Curve in Hough Space`: each clicked point becomes a line in `(m, b)` space.
- `Smooth Hough Votes`: a binned, smooth visualization of where those curves intersect.
- `Weighted Quadratic Loss`: the continuous objective minimized by the repo's quadratic Hough tracker.

Controls:

- Left click in `Point Space` to add a point.
- Right click in `Point Space` to remove the nearest point.
- `decay` makes older points matter less.
- `vote sigma` changes how wide each point's vote is in the visual heatmap.

## 1. Line Model

This project represents a camera line as:

\[
x = m y + b
\]

where:

- \(m\) is the slope
- \(b\) is the intercept

This is the same convention used in `perception/dvs_algorithms.py`.

## 2. What One Point Means in Hough Space

Suppose you click a point \((x_i, y_i)\).

Any line passing through that point must satisfy:

\[
x_i = m y_i + b
\]

Solving for \(b\):

\[
b = x_i - m y_i
\]

So one point in image space becomes one curve in parameter space. If several points lie on the same physical line, their curves intersect near the true \((m, b)\).

That is the core idea behind the Hough transform.

## 3. Classical Accumulator View

A simple Hough accumulator discretizes \((m, b)\) into bins and counts votes:

\[
H(m, b) = \sum_i \mathbf{1}\left( |x_i - (m y_i + b)| < \varepsilon \right)
\]

The demo uses a smooth version for visualization:

\[
H_\sigma(m, b) = \sum_i w_i \exp\left(
-\frac{(x_i - (m y_i + b))^2}{2 \sigma^2}
\right)
\]

where:

- \(\sigma\) is the `vote sigma` slider
- \(w_i\) is the age-dependent weight from the `decay` slider

This panel is for intuition: bright regions mean "many points agree that the line parameters are here."

## 4. The Continuous Quadratic Tracker Used Here

The repo's `PaperHoughLineAlgorithm` does **not** keep a large discrete accumulator.

Instead, it keeps a compact quadratic objective:

\[
J(m, b) = \sum_i w_i \left(x_i - m y_i - b\right)^2
\]

This is a weighted least-squares energy over all events seen so far.

Expanding the square:

\[
J(m, b)
= A m^2 + B m b + C b^2 + D m + E b + \text{const}
\]

with:

\[
A = \sum_i w_i y_i^2
\]

\[
B = 2 \sum_i w_i y_i
\]

\[
C = \sum_i w_i
\]

\[
D = -2 \sum_i w_i x_i y_i
\]

\[
E = -2 \sum_i w_i x_i
\]

Those are exactly the five accumulators stored by the algorithm:

```python
A += y * y
B += 2.0 * y
C += 1.0
D += -2.0 * x * y
E += -2.0 * x
```

with decay applied before new events are added.

## 5. Recovering the Best Line Analytically

To find the best line, set the gradient of \(J(m, b)\) to zero:

\[
\frac{\partial J}{\partial m} = 2 A m + B b + D = 0
\]

\[
\frac{\partial J}{\partial b} = B m + 2 C b + E = 0
\]

Solving the 2x2 system gives:

\[
q = 4AC - B^2
\]

\[
b = \frac{DB - 2AE}{q}
\]

\[
m = \frac{BE - 2CD}{q}
\]

These are the same formulas used by `PaperHoughLineAlgorithm.update()`.

The demo marks this analytic solution in the parameter-space plots and overlays the recovered line back in point space.

## 6. Why the Decay Matters

If the decay is \(\lambda \in (0, 1]\), older events receive smaller weights:

\[
w_i = \lambda^{k-i}
\]

for an event that arrived \(k-i\) updates ago.

Effects:

- `decay = 1.0`: all clicked points matter equally
- lower decay: recent points dominate
- lower decay is more responsive, but noisier

That matches the tradeoff described in `docs/TASKS/DVS_FULL_SIM.md`, where smaller decay values reduce lag for fast pencil motion.

## 7. Relation to the Existing Code

Relevant files already in this repo:

- `perception/dvs_algorithms.py`
  - `PaperHoughLineAlgorithm`: the quadratic event-driven tracker
- `perception/vision.py`
  - `SimEventCameraInterface.generate_events()`: synthetic event generation from a true line
- `benchmarks/benchmark_hough.py`
  - static and falling-pencil tests for the Hough tracker
- `docs/architecture.MD`
  - overview of the vision pipeline and line parameter conventions

## 8. Important Note About Centering

The real implementation centers camera coordinates before fitting:

\[
x' = x - c_x,\quad y' = y - c_y
\]

It solves for the intercept in centered coordinates, then converts back:

\[
b_{\text{pixel}} = b_{\text{center}} + c_x - m c_y
\]

The demo uses already-centered coordinates for clarity, so the displayed intercept is directly the fitted \(b\).

## 9. Takeaway

There are two useful ways to think about the algorithm:

1. **Geometric Hough view**: every point votes for a curve in parameter space, and the intersection is the line.
2. **Optimization view**: the repo stores a quadratic approximation of agreement and solves for its minimum in closed form.

The second view is what makes the project implementation compact and fast enough for event-driven updates.
