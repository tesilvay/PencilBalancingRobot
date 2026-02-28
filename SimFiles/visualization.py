import os
from datetime import datetime
from contextlib import nullcontext
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

class Visualizer3D:

    def __init__(self, history, dt, L=0.2, fps=60):

        self.history = history
        self.dt = dt
        self.L = L
        self.fps = fps
        self.frame_period = 1.0 / fps
        self.total_sim_time = history.shape[0] * dt

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(-0.15, 0.15)
        self.ax.set_ylim(-0.15, 0.15)
        self.ax.set_zlim(0, L)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.table_plot, = self.ax.plot([], [], [], 'k-', linewidth=2)
        self.pencil_plot, = self.ax.plot([], [], [], 'r-', linewidth=3)
        
        # --- Pencil tip trail ---
        self.trail_length = 120   # number of frames to keep
        self.tip_history = []

        self.trail_plot, = self.ax.plot(
            [], [], [],
            linestyle=':',
            linewidth=1,
            color='gray'
        )

        self.info_text = self.ax.text2D(
            0.02, 0.95,
            "",
            transform=self.ax.transAxes
        )

        plt.ion()
        plt.show()

    # -------------------------------------------------
    # Render a single frame
    # -------------------------------------------------
    def render_frame(self, state, sim_time):

        x = state[0]
        alpha_x = state[2]
        y = state[4]
        alpha_y = state[6]

        size = 0.05

        # --- Table ---
        corners_x = [x - size, x + size, x + size, x - size, x - size]
        corners_y = [y - size, y - size, y + size, y + size, y - size]
        corners_z = [0, 0, 0, 0, 0]

        self.table_plot.set_data(corners_x, corners_y)
        self.table_plot.set_3d_properties(corners_z)

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

        # --- Text ---
        self.info_text.set_text(
            f"Sim Time: {sim_time:.3f} s\n"
            f"x: {x:.3f}\n"
            f"y: {y:.3f}\n"
            f"αx: {alpha_x:.3f}\n"
            f"αy: {alpha_y:.3f}"
        )

        plt.draw()
        plt.pause(0.001)

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
            os.makedirs("simulation_video", exist_ok=True)

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_video/pencil_sim_{timestamp}.mp4"

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

                sim_time = frame_number * self.frame_period * video_speed

                if sim_time >= total_sim_time:
                    break

                sim_index = int(round(sim_time / self.dt))

                if sim_index >= len(self.history):
                    break

                state = self.history[sim_index]

                self.render_frame(state, sim_time)

                if save_video:
                    writer.grab_frame()

                frame_number += 1

        if save_video:
            print(f"Video saved to: {filename}")
        else:
            print("Playback finished.")
 