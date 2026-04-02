from .base        import BaseController
from .pole        import PolePlacementController,        PoleParams,        POLE_PRESETS
from .lqr         import LQRController,                  LQRParams,         LQR_PRESETS
from .smooth_pole import SmoothPolePlacementController,  SmoothPoleParams,  SMOOTH_POLE_PRESETS
from .circle      import CircleController,               CircleParams,      CIRCLE_PRESETS
from .null        import NullController
from src.shared     import Spec, NullParams, NULL_PRESETS
from src.system.plant import PLANT_REGISTRY

CONTROLLER_REGISTRY = {
    "pole":        Spec(PolePlacementController,       PoleParams,       POLE_PRESETS,        registries={"plant": PLANT_REGISTRY}),
    "lqr":         Spec(LQRController,                 LQRParams,        LQR_PRESETS,         registries={"plant": PLANT_REGISTRY}),
    "smooth_pole": Spec(SmoothPolePlacementController, SmoothPoleParams, SMOOTH_POLE_PRESETS, registries={"plant": PLANT_REGISTRY}),
    "circle":      Spec(CircleController,              CircleParams,     CIRCLE_PRESETS,      registries={"plant": PLANT_REGISTRY}),
    "null":        Spec(NullController,                NullParams,       NULL_PRESETS),
}
