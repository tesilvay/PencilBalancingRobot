import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class FiveBarVisualizer:

    def __init__(self, mech, workspace):
        self.mech = mech
        self.workspace = workspace

    def plot_workspace(self, points):

        fig, ax = plt.subplots()

        ax.set_aspect('equal')
        ax.grid(True)

        ax.scatter(points[:, 0], points[:, 1], s=5)

        # Choose alpha based on workspace span so the polygon wraps the points.
        span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
        alpha = 1.0 / max(span * 0.1, 1e-6)
        poly = self.workspace.alpha_shape(points, alpha=alpha)

        # If we get a MultiPolygon, take the largest component for visualization.
        if hasattr(poly, "geoms"):
            poly_main = max(poly.geoms, key=lambda g: g.area)
        else:
            poly_main = poly

        x, y = poly_main.exterior.xy
        ax.plot(x, y, 'k')

        center,r = self.workspace.largest_inscribed_circle(poly)

        if center:
            circ = plt.Circle(center,r,fill=False,color='red')
            ax.add_patch(circ)

        plt.show()

    def interactive_workspace(self, points):
        # Layout: main axes above, slider below (integer L = l_a = l_b, 70--200)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.7])
        ax.set_aspect('equal')
        ax.grid(True)
        # Fixed limits matching sweep range (same formula as workspace._cartesian_bounds, L_max=200)
        x_min, y_min, x_max, y_max = self._fixed_sweep_limits()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Adaptive sweep params for real-time updates (no progress bar)
        max_res = 70
        min_res = 10
        samples_per_cell = 3

        # ---- workspace points (updatable) ----
        scat = ax.scatter(points[:, 0], points[:, 1], s=6, color='red')

        # ---- workspace boundary (updatable) ----
        span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1])) if points.size else 1.0
        alpha = 1.0 / max(span * 0.1, 1e-6)
        poly = self.workspace.alpha_shape(points, alpha=alpha) if len(points) else None
        if hasattr(poly, "geoms") and poly is not None:
            poly_main = max(poly.geoms, key=lambda g: g.area)
        elif poly is not None:
            poly_main = poly
        else:
            poly_main = None
        if poly_main is not None and getattr(poly_main, "exterior", None) is not None:
            x, y = poly_main.exterior.xy
            boundary_line, = ax.plot(x, y, 'k', linewidth=2)
        else:
            boundary_line, = ax.plot([], [], 'k', linewidth=2)

        # ---- largest inscribed circle (updatable) ----
        center, radius = (None, 0.0) if poly is None else self.workspace.safe_workspace_circle(poly)
        diameter = 2 * radius if center is not None else 0.0
        legend_text = (
            f"Circle center: ({center[0]:.2f}, {center[1]:.2f})\n"
            f"Radius: {radius:.2f} mm\n"
            f"Diameter: {diameter:.2f} mm"
        ) if center is not None else "No workspace"
        text_artist = ax.text(
            0.98, 0.98, legend_text,
            transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )
        circle = plt.Circle((0, 0), 0, fill=False, color='green', linewidth=2)
        ax.add_patch(circle)
        center_dot = ax.scatter([], [], color='green')
        if center is not None:
            circle.center = center
            circle.set_radius(radius)
            center_dot.set_offsets(np.array([center]))
        else:
            circle.set_visible(False)
            center_dot.set_visible(False)

        # ---- slider: L (l_a = l_b), integers 70--200 ----
        slider_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
        L_min, L_max = 70, 200
        L_init = int(np.clip(round(self.mech.la), L_min, L_max))
        length_slider = Slider(
            slider_ax, "L (l_a = l_b)", L_min, L_max,
            valinit=L_init, valstep=1
        )

        # base joints and mechanism artists (end effector at circle center by default)
        O_g, B_g = self.mech.tf.bases_global()
        mech_artists = []

        def draw_mechanism_at(target_g, silent=False):
            """Draw the mechanism with end effector at target_g. Clears previous mech_artists."""
            nonlocal mech_artists
            for artist in mech_artists:
                artist.remove()
            mech_artists = []
            try:
                theta1, theta4, A_g, C_g, P_g = self.mech.solve(target_g)
                if not silent:
                    print("\n-----------------------------")
                    print(f"End effector at: ({target_g[0]:.3f}, {target_g[1]:.3f})")
                    print("Angles (deg):", np.degrees([theta1, theta4]))
                mech_artists += ax.plot([O_g[0], A_g[0]], [O_g[1], A_g[1]], 'b', linewidth=3)
                mech_artists += ax.plot([B_g[0], C_g[0]], [B_g[1], C_g[1]], 'm', linewidth=3)
                mech_artists += ax.plot([A_g[0], P_g[0]], [A_g[1], P_g[1]], 'b--', linewidth=2)
                mech_artists += ax.plot([C_g[0], P_g[0]], [C_g[1], P_g[1]], 'm--', linewidth=2)
                mech_artists.append(ax.scatter(P_g[0], P_g[1], color='green'))
            except Exception as e:
                if not silent:
                    print("IK failure:", e)

        # Initial mechanism: end effector at inscribed circle center (view already set to fixed sweep limits)
        if center is not None:
            draw_mechanism_at(center, silent=True)

        def update_workspace(_):
            L = int(length_slider.val)
            self.mech.la = self.mech.lb = L
            self.workspace._numba_constants = None
            pts = self.workspace.sweep_cartesian_adaptive(
                max_res=max_res, min_res=min_res, samples_per_cell=samples_per_cell,
                show_progress=False
            )
            # Update scatter
            if pts.size > 0:
                scat.set_offsets(pts)
                scat.set_visible(True)
                span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
                alpha_val = 1.0 / max(span * 0.1, 1e-6)
                poly_new = self.workspace.alpha_shape(pts, alpha=alpha_val)
                if hasattr(poly_new, "geoms"):
                    poly_main_new = max(poly_new.geoms, key=lambda g: g.area)
                else:
                    poly_main_new = poly_new
                ext = getattr(poly_main_new, "exterior", None)
                if ext is not None:
                    x, y = ext.xy
                    boundary_line.set_data(x, y)
                else:
                    boundary_line.set_data([], [])
                boundary_line.set_visible(True)
                cen, rad = self.workspace.safe_workspace_circle(poly_new)
                if cen is not None:
                    circle.center = cen
                    circle.set_radius(rad)
                    circle.set_visible(True)
                    center_dot.set_offsets(np.array([cen]))
                    center_dot.set_visible(True)
                    text_artist.set_text(
                        f"Circle center: ({cen[0]:.2f}, {cen[1]:.2f})\n"
                        f"Radius: {rad:.2f} mm\nDiameter: {2*rad:.2f} mm"
                    )
                    draw_mechanism_at(cen, silent=True)
                else:
                    circle.set_visible(False)
                    center_dot.set_visible(False)
                    text_artist.set_text("No inscribed circle")
            else:
                scat.set_offsets(np.empty((0, 2)))
                scat.set_visible(False)
                boundary_line.set_data([], [])
                boundary_line.set_visible(False)
                circle.set_visible(False)
                center_dot.set_visible(False)
                text_artist.set_text("No workspace")
            # Keep fixed x/y limits (no view change)
            fig.canvas.draw_idle()

        length_slider.on_changed(update_workspace)

        def on_click(event):
            nonlocal mech_artists
            if event.inaxes is None or event.inaxes != ax:
                return
            target_g = np.array([event.xdata, event.ydata])
            draw_mechanism_at(target_g, silent=False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_click)
        ax.set_title("Workspace Explorer")
        plt.show()

    def _workspace_bounds(self):
        """Bounding box for the current mechanism workspace (for axis limits)."""
        O_g, B_g = self.mech.tf.bases_global()
        la, lb = self.mech.la, self.mech.lb
        x_max = float(max(O_g[0], B_g[0])) + la + lb
        y_max = float(max(O_g[1], B_g[1])) + la + lb
        return 0.0, 0.0, x_max, y_max

    def _fixed_sweep_limits(self, L_max=200, margin=20):
        """Fixed axis limits matching the sweep range (first quadrant, max extent for L up to L_max)."""
        O_g, B_g = self.mech.tf.bases_global()
        x_max = float(max(O_g[0], B_g[0])) + L_max + L_max + margin
        y_max = float(max(O_g[1], B_g[1])) + L_max + L_max + margin
        return 0.0, 0.0, x_max, y_max