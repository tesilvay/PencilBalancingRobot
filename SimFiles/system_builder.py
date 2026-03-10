import numpy as np
from core.controller import NullController, PolePlacementController, LQRController
from perception.estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from perception.vision import SimVisionModel, RealEventCameraInterface
from core.model import BuildLinearModel
from fivebar.transform import FiveBarTransform
from fivebar.mechanism import FiveBarMechanism
from hardware.Servo_System import ServoSystem


def build_system(config, params, camera_params, x_ref=None):

    A, B = BuildLinearModel(params)

    # Controller
    if config.controller_type == "pole":
        poles = [-14 ,-16, -18, -20] * 2
        controller = PolePlacementController(A, B, poles, x_ref)

    elif config.controller_type == "lqr":
        Q_single_axis = np.diag([10, 0.1, 10, 1]) # x, x_dot, alpha, alpha_dot
        Z4 = np.zeros((4, 4))

        # Symmetric block diagonal for x and y axes
        Q = np.block([
            [Q_single_axis, Z4],
            [Z4, Q_single_axis]
        ])
        
        
        # Empirically, R > 1e5 keeps the poles in a reasonable range.
        R = np.eye(2) * 1e5
               
        
        controller = LQRController(A, B, Q, R, x_ref)

    else:
        controller = NullController()

    # Estimator
    estimator = None
    vision = None

    if config.estimator_type is not None:
        
        if params.dvs_cam:
            print("Using DVS camera")
            #vision = RealEventCameraInterface()
        else:
            vision = SimVisionModel(
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
    
    actuator = None
    if params.servo:
        actuator = ServoSystem(mech)
        
    

    return controller, vision, estimator, mech, actuator
