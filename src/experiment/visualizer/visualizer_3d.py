from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared import WorkspaceParams, default_workspace
from src.experiment.visualizer.visualizer3d import Visualizer3D


@dataclass
class Visualizer3DParams:
    L:            float = 0.15
    fps:          int = 60
    history:      np.ndarray | None = None
    dt:           float | None = None
    workspace:    WorkspaceParams = field(default_factory=default_workspace)
    mech:         Any = None
    mech_history: np.ndarray | None = None
    cmd_history:  np.ndarray | None = None


VISUALIZER_3D_PRESETS = {
    "default": {
        "L":   0.15,
        "fps": 60,
    }
}
