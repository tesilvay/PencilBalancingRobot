import numpy as np
from visualization.realtime_visualizer import PencilVisualizerRealtime
from core.controller import NullController, PolePlacementController, LQRController
from perception.estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from perception.vision import SimVisionModel, RealEventCameraInterface
from core.model import BuildLinearModel
from core.plant import BalancerPlant
from fivebar.transform import FiveBarTransform
from fivebar.mechanism import FiveBarMechanism
from hardware.Servo_System import ServoSystem
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SurfaceRegressionAlgorithm

def build_plant(params):
    plant = BalancerPlant(params)
    return plant

def build_controller(config, params, x_ref):
    A, B = BuildLinearModel(params)

    if config.controller_type == "pole":
        poles = [-14, -16, -18, -20] * 2
        controller = PolePlacementController(A, B, poles, x_ref)

    elif config.controller_type == "lqr":
        Q_single_axis = np.diag([10, 0.1, 10, 1])  # x, x_dot, alpha, alpha_dot
        Z4 = np.zeros((4, 4))

        # Symmetric block diagonal for x and y axes
        Q = np.block([
            [Q_single_axis, Z4],
            [Z4, Q_single_axis]
        ])

        # Empirically large R keeps poles reasonable
        R = np.eye(2) * 1e5

        controller = LQRController(A, B, Q, R, x_ref)

    else:
        controller = NullController()

    return controller

def build_estimator(config, params):
    A, B = BuildLinearModel(params)
    
    if config.estimator_type is not None:
        if config.estimator_type == "fd":
            estimator = FiniteDifferenceEstimator()

        elif config.estimator_type == "lpf":
            estimator = LowPassFiniteDifferenceEstimator()

        elif config.estimator_type == "kalman":
            Qk = np.eye(8) * 1e-6
            Rk = np.eye(4) * config.noise_std**2
            estimator = KalmanEstimator(A, dt=0.001, Q=Qk, R=Rk)
    else:
        estimator = None
    
    return estimator

def build_vision(config, params, camera_params):
    if config.estimator_type is not None:
        
        if params.dvs_cam:
            print("Using DVS camera")
            
            cam1_estimator = PaperHoughLineAlgorithm()
            cam2_estimator = PaperHoughLineAlgorithm()
            
            vision = RealEventCameraInterface(
                        camera_params=camera_params, 
                        cam1_estimator=cam1_estimator,
                        cam2_estimator=cam2_estimator,
            )
        else:
            vision = SimVisionModel(
                camera_params,
                noise_std=config.noise_std,
                delay_steps=config.delay_steps
            )
    else:
        vision = None

    return vision

def build_mechanism(params):

    if params.O is not None:

        tf = FiveBarTransform(params.O, params.B)

        mech = FiveBarMechanism(
            tf,
            la=params.la,
            lb=params.lb
        )
    else:
        mech = None
        
    return mech

def build_actuator(params, mech):

    if not params.servo:
        return None

    return ServoSystem(mech, port=params.servo_port)

def build_visualizer(params):
    if params.realtimerender:
        visualizer = PencilVisualizerRealtime()
    else:
        visualizer = None
    
    return visualizer

def build_system(config, params, camera_params, x_ref=None):

    plant = build_plant(params)

    controller = build_controller(config, params, x_ref)

    estimator = build_estimator(config, params)
    
    vision = build_vision(config, params, camera_params)
    
    mech = build_mechanism(params)
    
    actuator = build_actuator(params, mech)
    
    visualizer = build_visualizer(params)

    return plant, controller, vision, estimator, mech, actuator, visualizer
