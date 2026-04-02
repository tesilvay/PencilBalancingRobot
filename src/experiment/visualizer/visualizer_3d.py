from dataclasses import dataclass

from src.experiment.visualizer.visualizer3d import Visualizer3D


@dataclass
class Visualizer3DParams:
    L:   float
    fps: int


VISUALIZER_3D_PRESETS = {
    "default": {
        "L":   0.15,
        "fps": 60,
    }
}
