import numpy as np
from visualization.realtime_visualizer import PencilVisualizerRealtime
from core.controller import NullController, PolePlacementController, LQRController
from perception.estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from perception.vision import SimVisionModel, RealEventCameraInterface, SimEventCameraInterface
from core.model import BuildLinearModel
from core.plant import BalancerPlant
from core.sim_types import make_reference_state
from fivebar.transform import FiveBarTransform
from fivebar.mechanism import FiveBarMechanism
from hardware.Servo_System import ServoSystem
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SurfaceRegressionAlgorithm

def build_plant(params):
    plant = BalancerPlant(params)
    return plant

def build_controller(variant, params):
    A, B = BuildLinearModel(params)
    x_ref = make_reference_state(params.workspace)

    if variant.controller_type == "pole":
        poles = [-14, -16, -18, -20] * 2
        controller = PolePlacementController(A, B, poles, x_ref)

    elif variant.controller_type == "lqr":
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

def build_estimator(variant, params):
    A, B = BuildLinearModel(params)
    
    if variant.estimator_type is not None:
        if variant.estimator_type == "fd":
            estimator = FiniteDifferenceEstimator()

        elif variant.estimator_type == "lpf":
            estimator = LowPassFiniteDifferenceEstimator()

        elif variant.estimator_type == "kalman":
            Qk = np.eye(8) * 1e-6
            Rk = np.eye(4) * variant.noise_std**2
            estimator = KalmanEstimator(A, dt=0.001, Q=Qk, R=Rk)
    else:
        estimator = None
    
    return estimator

def dvs_cams_connected(params):
    hw = params.hardware
    return hw.dvs_cam_y_port is not None and hw.dvs_cam_x_port is not None

def build_vision(variant, params, camera_params):
    if variant.estimator_type is not None:
        hw = params.hardware
        if hw.dvs_cam:
            # decay=0.95: responsive to falling pencil (~1° median error vs 3.5° at 0.999)
            cam1_algo = PaperHoughLineAlgorithm(decay=0.95)
            cam2_algo = PaperHoughLineAlgorithm(decay=0.95)


            if dvs_cams_connected(params):
                vision = RealEventCameraInterface(
                    camera_params=camera_params,
                    cam1_algo=cam1_algo,
                    cam2_algo=cam2_algo,
                    cam1_serial=params.hardware.dvs_cam_x_port,
                    cam2_serial=params.hardware.dvs_cam_y_port,
                )
                
            else:
                vision = SimEventCameraInterface(
                            camera_params=camera_params, 
                            cam1_algo=cam1_algo,
                            cam2_algo=cam2_algo,
                )

        else:
            vision = SimVisionModel(
                camera_params,
                noise_std=variant.noise_std,
                delay_steps=variant.delay_steps
            )
    else:
        vision = None

    return vision

def build_mechanism(params):

    if params.mechanism is not None:
        m = params.mechanism
        tf = FiveBarTransform(m.O, m.B)

        mech = FiveBarMechanism(
            tf,
            la=m.la,
            lb=m.lb
        )
    else:
        mech = None
        
    return mech

def build_actuator(params, mech):

    if not params.hardware.servo:
        return None

    return ServoSystem(mech, port=params.hardware.servo_port)

def build_visualizer(params): #should know whether I have sim cam or real cam
    if params.run.realtimerender:
        visualizer = PencilVisualizerRealtime()
    else:
        visualizer = None
    
    return visualizer

def build_system(variant, params, camera_params):

    plant = build_plant(params)

    controller = build_controller(variant, params)

    estimator = build_estimator(variant, params)
    
    vision = build_vision(variant, params, camera_params)
    
    mech = build_mechanism(params)
    
    actuator = build_actuator(params, mech)
    
    visualizer = build_visualizer(params)

    return plant, controller, vision, estimator, mech, actuator, visualizer
