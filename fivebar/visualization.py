import matplotlib.pyplot as plt
import numpy as np


class FiveBarVisualizer:

    def __init__(self, mech, workspace):
        self.mech = mech
        self.workspace = workspace

    def plot_workspace(self, points):

        fig, ax = plt.subplots()

        ax.set_aspect('equal')
        ax.grid(True)

        ax.scatter(points[:,0], points[:,1], s=5)

        poly = self.workspace.alpha_shape(points, alpha=0.02)

        x,y = poly.exterior.xy
        ax.plot(x,y,'k')

        center,r = self.workspace.largest_inscribed_circle(poly)

        if center:
            circ = plt.Circle(center,r,fill=False,color='red')
            ax.add_patch(circ)

        plt.show()

    def interactive_workspace(self, points):

        fig, ax = plt.subplots()

        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(-75, 125)
        ax.set_ylim(-75, 145)

        # ---- workspace points ----
        # ax.scatter(points[:,0], points[:,1], s=6, color='red')

        # ---- workspace boundary ----
        poly = self.workspace.alpha_shape(points, alpha=0.02)

        x, y = poly.exterior.xy
        ax.plot(x, y, 'k', linewidth=2)

        # ---- largest inscribed circle ----
        center, radius = self.workspace.safe_workspace_circle(poly)
        
        # ---- show that in a legend ----
        diameter = 2 * radius

        legend_text = (
            f"Circle center: ({center[0]:.2f}, {center[1]:.2f})\n"
            f"Radius: {radius:.2f} mm\n"
            f"Diameter: {diameter:.2f} mm"
        )

        ax.text(
            0.98, 0.98,
            legend_text,
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        if center is not None:
            circle = plt.Circle(center, radius, fill=False, color='green', linewidth=2)
            ax.add_patch(circle)
            ax.scatter(center[0], center[1], color='green')

        # base joints
        O_g, B_g = self.mech.tf.bases_global()

        mech_artists = []

        def on_click(event):

            nonlocal mech_artists

            if event.inaxes is None:
                return

            # remove previous mechanism drawing
            for artist in mech_artists:
                artist.remove()
            mech_artists = []

            target = np.array([event.xdata, event.ydata])

            print("\n-----------------------------")
            print(f"Clicked: ({target[0]:.3f}, {target[1]:.3f})")

            try:

                theta1, theta4, A_l, C_l, P_l = self.mech.solve(target)

                A_g = self.mech.tf.l2g(A_l)
                C_g = self.mech.tf.l2g(C_l)
                P_g = self.mech.tf.l2g(P_l)

                print("Angles (deg):", np.degrees([theta1, theta4]), "point found:", P_g)

                # first links
                mech_artists += ax.plot(
                    [O_g[0], A_g[0]],
                    [O_g[1], A_g[1]],
                    'b', linewidth=3
                )

                mech_artists += ax.plot(
                    [B_g[0], C_g[0]],
                    [B_g[1], C_g[1]],
                    'm', linewidth=3
                )

                # second links
                mech_artists += ax.plot(
                    [A_g[0], P_g[0]],
                    [A_g[1], P_g[1]],
                    'b--', linewidth=2
                )

                mech_artists += ax.plot(
                    [C_g[0], P_g[0]],
                    [C_g[1], P_g[1]],
                    'm--', linewidth=2
                )

                mech_artists.append(
                    ax.scatter(P_g[0], P_g[1], color='green')
                )

                fig.canvas.draw_idle()

            except Exception as e:
                print("IK failure:", e)

        fig.canvas.mpl_connect('button_press_event', on_click)

        plt.title("Workspace Explorer")
        plt.show()