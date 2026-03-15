import os
from datetime import datetime
from contextlib import nullcontext
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

class Visualizer3D:

    def __init__(self, history, dt, L=0.2, fps=60, mech=None, mech_history=None, params=None, cmd_history=None):
        self.history = history
        self.dt = dt
        self.L = L
        self.fps = fps
        self.frame_period = 1.0 / fps
        self.total_sim_time = history.shape[0] * dt

        self.mech = mech
        self.mech_history = mech_history
        self.cmd_history = cmd_history  # (N, 2) table command at sim frequency, or None

        w = params.workspace
        self.x_ref = w.x_ref
        self.y_ref = w.y_ref
        self.safe_radius = w.safe_radius

        self.fig = plt.figure(figsize=(8, 8))   # width, height in inches
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(-0.15, 0.15)
        self.ax.set_ylim(-0.15, 0.15)
        self.ax.set_zlim(0, L)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.table_plot, = self.ax.plot([], [], [], 'k-', linewidth=2)
        self.table_cmd_plot, = self.ax.plot([], [], [], 'c--', linewidth=1)  # table command (current -> desired)
        self.pencil_plot, = self.ax.plot([], [], [], 'r-', linewidth=3)
        
        # Mechanism plot
        self.link_OA, = self.ax.plot([], [], [], 'b-', linewidth=3)
        self.link_AP, = self.ax.plot([], [], [], 'b--', linewidth=2)

        self.link_BC, = self.ax.plot([], [], [], 'm-', linewidth=3)
        self.link_CP, = self.ax.plot([], [], [], 'm--', linewidth=2)
        
        # --- Pencil tip trail ---
        self.trail_length = 120   # number of frames to keep
        self.tip_history = []

        self.trail_plot, = self.ax.plot(
            [], [], [],
            linestyle=':',
            linewidth=1,
            color='gray'
        )

        # --- Safe workspace circle ---
        if self.safe_radius is not None:

            theta = np.linspace(0, 2*np.pi, 200)

            circle_x = self.x_ref + self.safe_radius * np.cos(theta)
            circle_y = self.y_ref + self.safe_radius * np.sin(theta)
            circle_z = np.zeros_like(circle_x)

            self.safe_circle, = self.ax.plot(
                circle_x,
                circle_y,
                circle_z,
                linestyle=':',
                color='gray',
                linewidth=1.5
            )

        # --- Reference point ---
        self.ref_point, = self.ax.plot(
            [self.x_ref],
            [self.y_ref],
            [0],
            marker='*',
            markersize=3,
            color='black'
        )

        # Legend
        self.info_text = self.ax.text2D(
            0.02, 0.95,
            "",
            transform=self.ax.transAxes
        )

        # Disable 3D pan/rotate/zoom for faster redraws and fixed view
        self.ax.disable_mouse_rotation()

        plt.ion()
        plt.show()

    # -------------------------------------------------
    # Render a single frame
    # -------------------------------------------------
    def render_frame(self, state, sim_time, interactive=True):

        x = state[0]
        alpha_x = state[2]
        y = state[4]
        alpha_y = state[6]

        size = 0.01

        # --- Table ---
        corners_x = [x - size, x + size, x + size, x - size, x - size]
        corners_y = [y - size, y - size, y + size, y + size, y - size]
        corners_z = [0, 0, 0, 0, 0]

        self.table_plot.set_data(corners_x, corners_y)
        self.table_plot.set_3d_properties(corners_z)

        # --- Table command line (current position -> desired x_ref, y_ref) ---
        sim_index = int(round(sim_time / self.dt))
        if self.cmd_history is not None and 0 <= sim_index < len(self.cmd_history):
            cmd_x, cmd_y = self.cmd_history[sim_index, 0], self.cmd_history[sim_index, 1]
        else:
            cmd_x, cmd_y = self.x_ref, self.y_ref
        self.table_cmd_plot.set_data([x, cmd_x], [y, cmd_y])
        self.table_cmd_plot.set_3d_properties([0, 0])

        # --- Pencil ---
        base = np.array([x, y, 0.0])
        direction = np.array([alpha_x, alpha_y, 1.0])
        direction /= np.linalg.norm(direction)

        tip = base + direction * self.L

        self.pencil_plot.set_data(
            [base[0], tip[0]],
            [base[1], tip[1]]
        )
        self.pencil_plot.set_3d_properties(
            [base[2], tip[2]]
        )

        # --- Update Trail ---
        self.tip_history.append(tip)

        if len(self.tip_history) > self.trail_length:
            self.tip_history.pop(0)

        trail_array = np.array(self.tip_history)

        self.trail_plot.set_data(
            trail_array[:, 0],
            trail_array[:, 1]
        )
        self.trail_plot.set_3d_properties(
            trail_array[:, 2]
        )
        
        # --- Mechanism ---
        if self.mech is not None:

            scale = 1.0 / 1000.0
            O = np.array(self.mech.tf.O_g) * scale
            B = np.array(self.mech.tf.B_g) * scale

            if self.mech_history is not None:
                sim_index = int(round(sim_time / self.dt))
                if 0 <= sim_index < len(self.mech_history):
                    A_mm = self.mech_history[sim_index, 0, :]
                    C_mm = self.mech_history[sim_index, 1, :]
                    P_mm = self.mech_history[sim_index, 2, :]
                    if not (np.any(np.isnan(A_mm)) or np.any(np.isnan(C_mm)) or np.any(np.isnan(P_mm))):
                        A = A_mm * scale
                        C = C_mm * scale
                        P = P_mm * scale
                        self.link_OA.set_data([O[0], A[0]], [O[1], A[1]])
                        self.link_OA.set_3d_properties([0, 0])
                        self.link_AP.set_data([A[0], P[0]], [A[1], P[1]])
                        self.link_AP.set_3d_properties([0, 0])
                        self.link_BC.set_data([B[0], C[0]], [B[1], C[1]])
                        self.link_BC.set_3d_properties([0, 0])
                        self.link_CP.set_data([C[0], P[0]], [C[1], P[1]])
                        self.link_CP.set_3d_properties([0, 0])
            else:
                try:
                    target_mm = np.array([x, y]) * 1000.0
                    _, _, A_mm, C_mm, P_mm = self.mech.solve(target_mm)
                    A = A_mm * scale
                    C = C_mm * scale
                    P = P_mm * scale
                    self.link_OA.set_data([O[0], A[0]], [O[1], A[1]])
                    self.link_OA.set_3d_properties([0, 0])
                    self.link_AP.set_data([A[0], P[0]], [A[1], P[1]])
                    self.link_AP.set_3d_properties([0, 0])
                    self.link_BC.set_data([B[0], C[0]], [B[1], C[1]])
                    self.link_BC.set_3d_properties([0, 0])
                    self.link_CP.set_data([C[0], P[0]], [C[1], P[1]])
                    self.link_CP.set_3d_properties([0, 0])
                except Exception:
                    pass

        # --- Text (table command from history if available; cmd_x, cmd_y already set above) ---
        if self.cmd_history is None or not (0 <= sim_index < len(self.cmd_history)):
            cmd_x = getattr(self, "x_ref", 0.0)
            cmd_y = getattr(self, "y_ref", 0.0)
        self.info_text.set_text(
            f"Sim Time: {sim_time:.3f} s\n"
            f"x: {x:.3f}\n"
            f"y: {y:.3f}\n"
            f"αx: {alpha_x:.3f}\n"
            f"αy: {alpha_y:.3f}\n"
            f"Table cmd: ({cmd_x:.3f}, {cmd_y:.3f})"
        )

        if interactive:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            self.fig.canvas.draw()

    # -------------------------------------------------
    # Playback
    # -------------------------------------------------
    def render_video(self, video_speed=1.0, filename=None, save_video=True):
        """
        video_speed:
            1.0 → real time
            2.0 → 2x faster
            0.5 → half speed

        save_video:
            True  → saves MP4 file
            False → just plays animation live
        """

        if save_video:
            os.makedirs("simulation_videos", exist_ok=True)

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_videos/pencil_sim_{timestamp}.mp4"

            writer = FFMpegWriter(fps=self.fps)

        total_sim_time = self.history.shape[0] * self.dt

        print("Rendering...")

        frame_number = 0

        if save_video:
            context = writer.saving(self.fig, filename, dpi=100)
        else:
            context = nullcontext()
    
        with context:

            while True:

                # Stop if window was closed
                if not plt.fignum_exists(self.fig.number):
                    print("Window closed. Stopping simulation.")
                    break
    
                sim_time = frame_number * self.frame_period * video_speed

                if sim_time >= total_sim_time:
                    break

                sim_index = int(round(sim_time / self.dt))

                if sim_index >= len(self.history):
                    break

                state = self.history[sim_index]

                self.render_frame(state, sim_time, interactive=not save_video)

                if save_video:
                    writer.grab_frame()

                frame_number += 1

        if save_video:
            print(f"Video saved to: {filename}")
        else:
            print("Playback finished.")
 