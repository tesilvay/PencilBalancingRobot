from dataclasses import dataclass
import numpy as np








# ── Visualizers ───────────────────────────────────────────────

@dataclass
class SimDvsVisualizerParams:
    width:  int
    height: int

@dataclass
class RealDvsVisualizerParams:
    width:       int
    height:      int
    mask_y_cam1: int
    mask_y_cam2: int

@dataclass
class OneDvsVisualizerParams:
    cam_index:    int
    width:        int
    height:       int
    surface_gain: float

@dataclass
class SimDvsWorkspaceVisualizerParams:
    width:  int
    height: int

@dataclass
class RealDvsWorkspaceVisualizerParams:
    width:       int
    height:      int
    mask_y_cam1: int
    mask_y_cam2: int

@dataclass
class Visualizer3DParams:
    L:   float
    fps: int




# ── Experiment ────────────────────────────────────────────────

@dataclass
class ExperimentParams:
    system:         object
    logger:         object
    stop_condition: object
    visualizer:     dict
    progress:       object
    pacing:         object
    scheduler:      object

