import numpy as np
from core.controller import NullController, PolePlacementController, LQRController, build_lqr_weights
from perception.estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from perception.vision import VisionSystem
from core.model import BuildLinearModel
from fivebar.transform import FiveBarTransform
from fivebar.mechanism import FiveBarMechanism


def build_system(config, params, camera_params, x_ref=None):

    A, B = BuildLinearModel(params)

    # Controller
    if config.controller_type == "pole":
        poles = [-14 ,-16, -18, -20] * 2
        controller = PolePlacementController(A, B, poles, x_ref)

    elif config.controller_type == "lqr":
        Q, R = build_lqr_weights(
            x_max=0.05,
            xdot_max=0.5,
            alpha_max=0.2,
            alphadot_max=2.0,
            u_max=0.05,
            angle_importance=config.angle_importance,
            effort_scale=config.effort_scale
        )
        controller = LQRController(A, B, Q, R, x_ref)

    else:
        controller = NullController()

    # Estimator
    estimator = None
    vision = None

    if config.estimator_type is not None:

        vision = VisionSystem(
            camera_params,
            noise_std=config.noise_std,
            delay_steps=config.delay_steps
        )

        if config.estimator_type == "fd":
            estimator = FiniteDifferenceEstimator()

        elif config.estimator_type == "lpf":
            estimator = LowPassFiniteDifferenceEstimator()

        elif config.estimator_type == "kalman":
            Qk = np.eye(8) * 1e-6
            Rk = np.eye(4) * config.noise_std**2
            estimator = KalmanEstimator(A, dt=0.001, Q=Qk, R=Rk)
    
    # --- Five-bar mechanism ---
    mech = None

    if params.O is not None:

        tf = FiveBarTransform(params.O, params.B)

        mech = FiveBarMechanism(
            tf,
            la=params.la,
            lb=params.lb
        )

    return controller, vision, estimator, mech
