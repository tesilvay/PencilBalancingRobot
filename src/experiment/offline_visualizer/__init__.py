from src.shared import Spec, NULL_PRESETS, NullParams
from src.system.actuator.mech import MECHANISM_REGISTRY

from .base import OfflineVisualizerBase
from .null import NullOfflineVisualizer
from .visualizer3d import Visualizer3D, Visualizer3DParams, VISUALIZER_3D_PRESETS

OFFLINE_VISUALIZER_REGISTRY = {
    "null": Spec(NullOfflineVisualizer, NullParams, NULL_PRESETS),
    "3d": Spec(
        Visualizer3D,
        Visualizer3DParams,
        VISUALIZER_3D_PRESETS,
        registries={"mech": MECHANISM_REGISTRY},
    ),
}
