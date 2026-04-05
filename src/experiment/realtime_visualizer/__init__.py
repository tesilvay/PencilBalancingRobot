from src.shared import Spec, NULL_PRESETS, NullParams

from .base import RealtimeVisualizerBase, VizResult, WorkspacePanelRenderer
from .null import NullRealtimeVisualizer
from .sim_dvs import SimDvsVisualizer, SimDvsVisualizerParams, SIM_DVS_VISUALIZER_PRESETS
from .real_dvs import RealDvsVisualizer, RealDvsVisualizerParams, REAL_DVS_VISUALIZER_PRESETS
from .one_dvs import OneDvsVisualizer, OneDvsVisualizerParams, ONE_DVS_VISUALIZER_PRESETS
from .sim_dvs_workspace import (
    SimDvsWorkspaceVisualizer,
    SimDvsWorkspaceVisualizerParams,
    SIM_DVS_WORKSPACE_VISUALIZER_PRESETS,
)
from .real_dvs_workspace import (
    RealDvsWorkspaceVisualizer,
    RealDvsWorkspaceVisualizerParams,
    REAL_DVS_WORKSPACE_VISUALIZER_PRESETS,
)

REALTIME_VISUALIZER_REGISTRY = {
    "null": Spec(NullRealtimeVisualizer, NullParams, NULL_PRESETS),
    "sim": Spec(SimDvsVisualizer, SimDvsVisualizerParams, SIM_DVS_VISUALIZER_PRESETS),
    "real": Spec(RealDvsVisualizer, RealDvsVisualizerParams, REAL_DVS_VISUALIZER_PRESETS),
    "one": Spec(OneDvsVisualizer, OneDvsVisualizerParams, ONE_DVS_VISUALIZER_PRESETS),
    "sim_ws": Spec(SimDvsWorkspaceVisualizer, SimDvsWorkspaceVisualizerParams, SIM_DVS_WORKSPACE_VISUALIZER_PRESETS),
    "real_ws": Spec(RealDvsWorkspaceVisualizer, RealDvsWorkspaceVisualizerParams, REAL_DVS_WORKSPACE_VISUALIZER_PRESETS),
}
