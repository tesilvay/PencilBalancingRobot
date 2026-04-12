from .base        import BaseController
from .accel_pole  import AccelPolePlacementController,  AccelPoleParams,  ACCEL_POLE_PRESETS
from .accel_lqr   import AccelLQRController,            AccelLQRParams,   ACCEL_LQR_PRESETS
from .accel_lag_pole import AccelLagPolePlacementController, AccelLagPoleParams, ACCEL_LAG_POLE_PRESETS
from .pole        import PolePlacementController,       PoleParams,       POLE_PRESETS
from .lqr         import LQRController,                 LQRParams,        LQR_PRESETS
from .delta_lqr   import DeltaLQRController,            DeltaLQRParams,   DELTA_LQR_PRESETS
from .smooth_pole_cmd_state import (
    SmoothPoleCommandStateController,
    SmoothPoleCommandStateParams,
    SMOOTH_POLE_CMD_STATE_PRESETS,
)
from .smooth_pole import SmoothPolePlacementController, SmoothPoleParams, SMOOTH_POLE_PRESETS
from .smooth_lqr import SmoothLQRController, SmoothLQRParams, SMOOTH_LQR_PRESETS
from .smooth_pole_integral import (
    SmoothPoleIntegralController,
    SmoothPoleIntegralParams,
    SMOOTH_POLE_INTEGRAL_PRESETS,
)
from .simple_controller import SimpleController, SimpleControllerParams, SIMPLE_CONTROLLER_PRESETS
from .circle      import CircleController,              CircleParams,     CIRCLE_PRESETS
from .null        import NullController
from src.shared   import Spec, NullParams, NULL_PRESETS

CONTROLLER_REGISTRY = {
    "accel_pole":  Spec(AccelPolePlacementController,  AccelPoleParams,  ACCEL_POLE_PRESETS),
    "accel_lqr":   Spec(AccelLQRController,            AccelLQRParams,   ACCEL_LQR_PRESETS),
    "accel_lag_pole": Spec(AccelLagPolePlacementController, AccelLagPoleParams, ACCEL_LAG_POLE_PRESETS),
    "pole":        Spec(PolePlacementController,       PoleParams,       POLE_PRESETS),
    "lqr":         Spec(LQRController,                 LQRParams,        LQR_PRESETS),
    "delta_lqr":   Spec(DeltaLQRController,            DeltaLQRParams,   DELTA_LQR_PRESETS),
    "smooth_pole_cmd_state": Spec(
        SmoothPoleCommandStateController,
        SmoothPoleCommandStateParams,
        SMOOTH_POLE_CMD_STATE_PRESETS,
    ),
    "smooth_pole": Spec(SmoothPolePlacementController, SmoothPoleParams, SMOOTH_POLE_PRESETS),
    "smooth_lqr": Spec(SmoothLQRController, SmoothLQRParams, SMOOTH_LQR_PRESETS),
    "smooth_pole_integral": Spec(
        SmoothPoleIntegralController,
        SmoothPoleIntegralParams,
        SMOOTH_POLE_INTEGRAL_PRESETS,
    ),
    "simple":      Spec(SimpleController,               SimpleControllerParams, SIMPLE_CONTROLLER_PRESETS),
    "circle":      Spec(CircleController,              CircleParams,     CIRCLE_PRESETS),
    "null":        Spec(NullController,                NullParams,       NULL_PRESETS),
}
