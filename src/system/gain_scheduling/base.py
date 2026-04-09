from __future__ import annotations

import numpy as np

from src.shared import Measurement


class GainScheduler:
    def apply(self, y_raw: Measurement) -> Measurement:
        raise NotImplementedError

    def map_angle(self, angle_rad: float) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def plot_angle_shape(
        self,
        min_angle_deg: float = 0.0,
        max_angle_deg: float = 15.0,
        num_points: int = 401,
        *,
        ax=None,
        show: bool = True,
    ):
        import matplotlib.pyplot as plt

        in_deg = np.linspace(float(min_angle_deg), float(max_angle_deg), int(num_points))
        in_rad = np.deg2rad(in_deg)
        out_deg = np.rad2deg([self.map_angle(float(a)) for a in in_rad])

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(in_deg, out_deg, label=self.__class__.__name__)
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.axvline(0.0, color="0.7", linewidth=1.0)
        ax.set_xlabel("Measured angle (deg)")
        ax.set_ylabel("Mapped angle (deg)")
        ax.set_title("Gain schedule angle mapping")
        ax.set_xlim(0.0, 15.0)
        ax.set_ylim(0.0, 15.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if show:
            plt.show()

        return ax
