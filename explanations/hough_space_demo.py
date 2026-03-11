"""Interactive demo of Hough space voting and the repo's quadratic tracker.

Run:
    python explanations/hough_space_demo.py

Controls:
    - Left click in the point-space panel to add a point.
    - Right click in the point-space panel to remove the nearest point.
    - Use the sliders to change decay and vote width.
    - Use Undo / Reset to manage the point set.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


@dataclass
class FitResult:
    slope: float | None
    intercept: float | None
    q: float
    coeffs: tuple[float, float, float, float, float]


class QuadraticHoughDemo:
    """Interactive visualization for point-space and slope/intercept Hough space."""

    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []

        self.x_range = (-10.0, 10.0)
        self.y_range = (-10.0, 10.0)
        self.m_range = (-3.0, 3.0)
        self.b_range = (-12.0, 12.0)

        self.curve_m = np.linspace(*self.m_range, 400)
        self.grid_size = 220
        self.m_grid = np.linspace(*self.m_range, self.grid_size)
        self.b_grid = np.linspace(*self.b_range, self.grid_size)
        self.M, self.B = np.meshgrid(self.m_grid, self.b_grid)

        self.fig = plt.figure(figsize=(14, 9))
        grid = self.fig.add_gridspec(
            2,
            2,
            left=0.06,
            right=0.97,
            top=0.92,
            bottom=0.22,
            hspace=0.28,
            wspace=0.24,
        )

        self.ax_points = self.fig.add_subplot(grid[0, 0])
        self.ax_curves = self.fig.add_subplot(grid[0, 1])
        self.ax_votes = self.fig.add_subplot(grid[1, 0])
        self.ax_loss = self.fig.add_subplot(grid[1, 1])

        self._style_axes()
        self._build_controls()

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.update_visuals()

    def _style_axes(self) -> None:
        self.ax_points.set_title("Point Space")
        self.ax_points.set_xlabel("x")
        self.ax_points.set_ylabel("y")
        self.ax_points.set_xlim(*self.x_range)
        self.ax_points.set_ylim(*self.y_range)
        self.ax_points.grid(True, alpha=0.3)
        self.ax_points.set_aspect("equal")

        self.ax_curves.set_title("Each Point Becomes a Curve in Hough Space")
        self.ax_curves.set_xlabel("slope m")
        self.ax_curves.set_ylabel("intercept b")
        self.ax_curves.set_xlim(*self.m_range)
        self.ax_curves.set_ylim(*self.b_range)
        self.ax_curves.grid(True, alpha=0.3)

        self.ax_votes.set_title("Smooth Hough Votes")
        self.ax_votes.set_xlabel("slope m")
        self.ax_votes.set_ylabel("intercept b")

        self.ax_loss.set_title("Weighted Quadratic Loss")
        self.ax_loss.set_xlabel("slope m")
        self.ax_loss.set_ylabel("intercept b")

    def _build_controls(self) -> None:
        self.info_text = self.fig.text(
            0.06,
            0.13,
            "",
            fontsize=10,
            family="monospace",
            va="top",
        )

        decay_ax = self.fig.add_axes([0.10, 0.07, 0.30, 0.03])
        sigma_ax = self.fig.add_axes([0.48, 0.07, 0.30, 0.03])
        undo_ax = self.fig.add_axes([0.82, 0.06, 0.06, 0.05])
        reset_ax = self.fig.add_axes([0.90, 0.06, 0.06, 0.05])

        self.decay_slider = Slider(
            ax=decay_ax,
            label="decay",
            valmin=0.80,
            valmax=1.00,
            valinit=1.00,
            valstep=0.005,
        )
        self.sigma_slider = Slider(
            ax=sigma_ax,
            label="vote sigma",
            valmin=0.15,
            valmax=1.50,
            valinit=0.50,
            valstep=0.05,
        )
        self.undo_button = Button(undo_ax, "Undo")
        self.reset_button = Button(reset_ax, "Reset")

        self.decay_slider.on_changed(lambda _: self.update_visuals())
        self.sigma_slider.on_changed(lambda _: self.update_visuals())
        self.undo_button.on_clicked(self.on_undo)
        self.reset_button.on_clicked(self.on_reset)

    def _weights(self, count: int) -> np.ndarray:
        if count == 0:
            return np.zeros(0, dtype=float)
        decay = float(self.decay_slider.val)
        ages = np.arange(count - 1, -1, -1, dtype=float)
        return decay ** ages

    def _fit_quadratic_tracker(self) -> FitResult:
        decay = float(self.decay_slider.val)

        A = 0.0
        B = 0.0
        C = 0.0
        D = 0.0
        E = 0.0

        for x, y in self.points:
            A *= decay
            B *= decay
            C *= decay
            D *= decay
            E *= decay

            A += y * y
            B += 2.0 * y
            C += 1.0
            D += -2.0 * x * y
            E += -2.0 * x

        q = 4.0 * A * C - B * B
        if abs(q) < 1e-9:
            return FitResult(None, None, q, (A, B, C, D, E))

        intercept = (D * B - 2.0 * A * E) / q
        slope = (B * E - 2.0 * C * D) / q
        return FitResult(slope, intercept, q, (A, B, C, D, E))

    def _compute_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.points:
            empty = np.zeros_like(self.M)
            return empty, empty

        sigma = float(self.sigma_slider.val)
        weights = self._weights(len(self.points))

        votes = np.zeros_like(self.M)
        loss = np.zeros_like(self.M)

        for weight, (x, y) in zip(weights, self.points):
            residual = x - (self.M * y + self.B)
            votes += weight * np.exp(-0.5 * (residual / sigma) ** 2)
            loss += weight * (residual ** 2)

        return votes, loss

    def on_click(self, event) -> None:
        if event.inaxes != self.ax_points or event.xdata is None or event.ydata is None:
            return

        point = (float(event.xdata), float(event.ydata))

        if event.button == 1:
            self.points.append(point)
        elif event.button == 3 and self.points:
            pts = np.asarray(self.points)
            distances = np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])
            self.points.pop(int(np.argmin(distances)))
        else:
            return

        self.update_visuals()

    def on_undo(self, _event) -> None:
        if self.points:
            self.points.pop()
            self.update_visuals()

    def on_reset(self, _event) -> None:
        self.points.clear()
        self.update_visuals()

    def update_visuals(self) -> None:
        votes, loss = self._compute_surfaces()
        fit = self._fit_quadratic_tracker()

        self.ax_points.clear()
        self.ax_curves.clear()
        self.ax_votes.clear()
        self.ax_loss.clear()
        self._style_axes()

        self.ax_votes.imshow(
            votes,
            origin="lower",
            extent=[*self.m_range, *self.b_range],
            aspect="auto",
            cmap="viridis",
        )
        self.ax_loss.imshow(
            loss,
            origin="lower",
            extent=[*self.m_range, *self.b_range],
            aspect="auto",
            cmap="magma_r",
        )

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.points), 1)))

        if self.points:
            pts = np.asarray(self.points)
            self.ax_points.scatter(pts[:, 0], pts[:, 1], c=colors[: len(pts)], s=45)

            for index, ((x, y), color) in enumerate(zip(self.points, colors), start=1):
                curve_b = x - self.curve_m * y
                self.ax_curves.plot(
                    self.curve_m,
                    curve_b,
                    color=color,
                    alpha=0.85,
                    linewidth=1.5,
                    label=f"p{index}: ({x:.1f}, {y:.1f})",
                )

            if len(self.points) <= 8:
                self.ax_curves.legend(loc="upper right", fontsize=8)

        if fit.slope is not None and fit.intercept is not None:
            ys = np.linspace(*self.y_range, 200)
            xs = fit.slope * ys + fit.intercept

            self.ax_points.plot(xs, ys, color="crimson", linewidth=2.5)
            self.ax_points.scatter(
                [fit.intercept],
                [0.0],
                color="crimson",
                marker="x",
                s=80,
            )

            self.ax_curves.scatter(
                [fit.slope],
                [fit.intercept],
                color="crimson",
                marker="x",
                s=80,
            )
            self.ax_votes.scatter(
                [fit.slope],
                [fit.intercept],
                color="white",
                marker="x",
                s=80,
            )
            self.ax_loss.scatter(
                [fit.slope],
                [fit.intercept],
                color="cyan",
                marker="x",
                s=80,
            )

        if self.points:
            peak_index = np.unravel_index(np.argmax(votes), votes.shape)
            peak_b = self.b_grid[peak_index[0]]
            peak_m = self.m_grid[peak_index[1]]

            self.ax_votes.scatter(
                [peak_m],
                [peak_b],
                color="red",
                marker="o",
                s=35,
                facecolors="none",
                linewidths=1.5,
            )

        A, B, C, D, E = fit.coeffs
        line_text = "line unavailable"
        if fit.slope is not None and fit.intercept is not None:
            line_text = f"x = {fit.slope:+.3f} y {fit.intercept:+.3f}"

        self.info_text.set_text(
            "Left click: add point    Right click: remove nearest point\n"
            "The red or cyan marker is the analytic minimum of the same quadratic used in "
            "PaperHoughLineAlgorithm.\n"
            f"points={len(self.points)}    {line_text}    q={fit.q:.4f}\n"
            f"A={A:.3f}    B={B:.3f}    C={C:.3f}    D={D:.3f}    E={E:.3f}"
        )

        self.fig.canvas.draw_idle()

    def show(self) -> None:
        self.fig.suptitle("Hough Space Demo: point votes, accumulator intuition, quadratic fit")
        plt.show()


def main() -> None:
    demo = QuadraticHoughDemo()
    demo.show()


if __name__ == "__main__":
    main()
