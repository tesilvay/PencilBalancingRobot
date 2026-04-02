from .base         import VisionModelBase
from .sim_analytic import SimVisionModel,            SimAnalyticParams, SIM_ANALYTIC_PRESETS
from .sim_dvs      import SimEventCameraInterface,   SimDVSParams,      SIM_DVS_PRESETS
from .real_dvs     import RealEventCameraInterface,  RealDVSParams,     REAL_DVS_PRESETS
from src.shared    import Spec

VISION_INTERFACE_REGISTRY = {
    "sim_analytic": Spec(SimVisionModel,            SimAnalyticParams, SIM_ANALYTIC_PRESETS, sim_only=True),
    "sim_dvs":      Spec(SimEventCameraInterface,   SimDVSParams,      SIM_DVS_PRESETS,      sim_only=True),
    "real_dvs":     Spec(RealEventCameraInterface,  RealDVSParams,     REAL_DVS_PRESETS,     sim_only=False),
}
