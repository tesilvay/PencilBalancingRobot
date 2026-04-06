from __future__ import annotations

import os
import time
from datetime import datetime
from contextlib import nullcontext
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from dataclasses import dataclass, field
from typing import Any

from src.shared import ControlInput, WorkspaceParams, default_workspace

from .base import OfflineVisualizerBase
from src.experiment.logger.logger import SimulationResult


@dataclass
class Visualizer3DParams:
    L:            float
    fps:          int
    add_trail:    bool = True
    history:      np.ndarray | None = None
    dt:           float | None = None
    workspace:    WorkspaceParams = field(default_factory=default_workspace)
    mech:         Any = None
    mech_history: np.ndarray | None = None
    cmd_history:  np.ndarray | None = None


VISUALIZER_3D_PRESETS = {
    "default": {
        "L":         0.15,
        "fps":       24,
        "add_trail": True,
        "mech":      "five_bar:default",
    }
}



class Visualizer3D(OfflineVisualizerBase):

    def __init__(self, params: Visualizer3DParams):

        self.history = params.history
        self.dt = params.dt
        self.L = params.L
        self.fps = params.fps
        self.frame_period = 1.0 / params.fps

        if params.history is not None and params.dt is not None:
            self.total_sim_time = params.history.shape[0] * params.dt
        else:
            self.total_sim_time = 0.0

        self.mech = params.mech
        self.mech_history = params.mech_history
        # Logged actuator geometry (IK at commanded setpoint); kept for debugging only.
        self.mech_history_cmd: np.ndarray | None = None
        self.cmd_history = params.cmd_history
        self.add_trail = params.add_trail

        w = params.workspace
        self.x_ref = w.x_ref
        self.y_ref = w.y_ref
        self.safe_radius = w.safe_radius

        self.fig = plt.figure(figsize=(8, 8))   # width, height in inches
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(-0.15, 0.15)
        self.ax.set_ylim(-0.15, 0.15)
        self.ax.set_zlim(0, self.L)

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
        self.trail_length = 20   # number of frames to keep
        self.tip_history = []

        self.trail_plot, = self.ax.plot(
            [], [], [],
            linestyle=':',
            linewidth=1,
            color='gray'
        )
        self.trail_plot.set_visible(params.add_trail)

        # --- Safe workspace circle ---
        if self.safe_radius is not None:

            theta = np.linspace(0, 2*np.pi, 20)

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

    def _clear_mech_links(self) -> None:
        for link in (self.link_OA, self.link_AP, self.link_BC, self.link_CP):
            link.set_data([], [])
            link.set_3d_properties([])

    def _build_mech_history_from_state(self) -> None:
        """Fill ``mech_history`` with IK at true table pose (px, py) per frame.

        Uses the same workspace frame as :meth:`Mechanism.command_geometry` so the
        five-bar matches the pencil base. Logged ``mech_history_cmd`` remains the
        command-based geometry for debugging.
        """
        if self.mech is None or self.history is None or self.history.size == 0:
            self.mech_history = None
            return
        n = int(self.history.shape[0])
        out = np.full((n, 3, 2), np.nan, dtype=float)
        for i in range(n):
            row = self.history[i]
            if row.shape[0] < 8:
                continue
            px, py = float(row[0]), float(row[4])
            try:
                joints, _ = self.mech.command_geometry(
                    ControlInput(px_cmd=px, py_cmd=py)
                )
                out[i, :, :] = np.asarray(joints, dtype=float).reshape(3, 2)
            except (ValueError, TypeError):
                pass
        self.mech_history = out

    def _ensure_mech_history_aligned(self) -> None:
        """Precompute mechanism joints if ``render_video`` runs without ``finalize``."""
        if self.mech is None or self.history is None or self.history.size == 0:
            return
        n = int(self.history.shape[0])
        need = (
            self.mech_history is None
            or self.mech_history.ndim != 3
            or int(self.mech_history.shape[0]) != n
        )
        if need:
            self._build_mech_history_from_state()

    def finalize(self, result: SimulationResult, *, dt: float) -> None:
        """Load trajectories from a trial and play back interactively (no file write)."""
        self.history = np.asarray(result.state_history, dtype=float)
        self.dt = float(dt)
        self.tip_history = []
        if result.cmd_history is not None and len(result.cmd_history) > 0:
            self.cmd_history = np.asarray(result.cmd_history, dtype=float)
        else:
            self.cmd_history = None
        if result.mech_history is not None and len(result.mech_history) > 0:
            self.mech_history_cmd = np.asarray(result.mech_history, dtype=float)
        else:
            self.mech_history_cmd = None
        self.mech_history = None
        if self.history.size == 0:
            return
        self.total_sim_time = float(self.history.shape[0] * self.dt)
        if self.mech is not None:
            self._build_mech_history_from_state()
        self.render_video(save_video=False)

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

        # --- Trail (optional; skip work when disabled) ---
        if self.add_trail:
            self.tip_history.append(tip)
            if len(self.tip_history) > self.trail_length:
                self.tip_history.pop(0)
            trail_array = np.array(self.tip_history)
            self.trail_plot.set_data(
                trail_array[:, 0],
                trail_array[:, 1],
            )
            self.trail_plot.set_3d_properties(trail_array[:, 2])
        else:
            self.tip_history.clear()
            self.trail_plot.set_data([], [])
            self.trail_plot.set_3d_properties([])

        # --- Mechanism (joints from true-state IK; precomputed before playback) ---
        if self.mech is not None and self.mech_history is not None:
            scale = 1.0 / 1000.0
            O = np.array(self.mech.tf.O_g) * scale
            B = np.array(self.mech.tf.B_g) * scale
            sim_index_m = int(round(sim_time / self.dt))
            if 0 <= sim_index_m < len(self.mech_history):
                A_mm = self.mech_history[sim_index_m, 0, :]
                C_mm = self.mech_history[sim_index_m, 1, :]
                P_mm = self.mech_history[sim_index_m, 2, :]
                if not (
                    np.any(np.isnan(A_mm))
                    or np.any(np.isnan(C_mm))
                    or np.any(np.isnan(P_mm))
                ):
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
                    self._clear_mech_links()
            else:
                self._clear_mech_links()
        elif self.mech is not None:
            self._clear_mech_links()

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
            self.fig.canvas.flush_events()
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
            False → plays live, pacing each frame to wall time (1/fps)/video_speed;
                    if rendering is slower, no extra sleep (smooth but may lag real time).
        """

        if save_video:
            os.makedirs("simulation_videos", exist_ok=True)

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_videos/pencil_sim_{timestamp}.mp4"

            writer = FFMpegWriter(fps=self.fps)

        total_sim_time = self.history.shape[0] * self.dt

        self._ensure_mech_history_aligned()

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

                if save_video:
                    self.render_frame(state, sim_time, interactive=False)
                    writer.grab_frame()
                else:
                    wall_period = self.frame_period / video_speed
                    t0 = time.perf_counter()
                    self.render_frame(state, sim_time, interactive=True)
                    elapsed = time.perf_counter() - t0
                    sleep_for = wall_period - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                frame_number += 1

        if save_video:
            print(f"Video saved to: {filename}")
        else:
            print("Playback finished.")
 