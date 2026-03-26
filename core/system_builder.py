# system_builder.py
import numpy as np
from warnings import warn
from visualization.realtime_visualizer import PencilVisualizerRealtime, DVSWorkspaceVisualizer
from core.controller import NullController, PolePlacementController, LQRController, CircleController
from perception.estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from perception.vision import SimVisionModel, RealEventCameraInterface, SimEventCameraInterface, Perception
from core.model import BuildLinearModel
from core.plant import BalancerPlant
from core.sim_types import make_reference_state, StopPolicy
from fivebar.transform import FiveBarTransform
from fivebar.mechanism import FiveBarMechanism
from hardware.Servo_System import ServoSystem
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SamLineAlgorithm, SurfaceRegressionAlgorithm
from simulation.stop_conditions import MaxSteps, FallCondition, StabilizedCondition, AnyStop, Infinite
from simulation.pacing import NoPacing, RealTimePacing
from simulation.logger import Logger
from simulation.system import System
from simulation.scheduler import Scheduler
from simulation.runner import ExperimentRunner


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
        Q_single_axis = np.diag([0.01, 0.01, 100, 10])  # x, x_dot, alpha, alpha_dot
        Z4 = np.zeros((4, 4))

        # Symmetric block diagonal for x and y axes
        Q = np.block([
            [Q_single_axis, Z4],
            [Z4, Q_single_axis]
        ])

        # Empirically large R keeps poles reasonable
        R = np.eye(2) * 1e6

        controller = LQRController(A, B, Q, R, x_ref)

    elif variant.controller_type == "circle":
        radius = params.workspace.safe_radius
        period_s = 18
        controller = CircleController(x_ref, radius, period_s)
    else:
        controller = NullController()

    return controller

def build_estimator(variant, params):
    A, B = BuildLinearModel(params)
    
    if variant.estimator_type is not None:
        if variant.estimator_type in ("fd", "fde"):
            estimator = FiniteDifferenceEstimator()

        elif variant.estimator_type == "lpf":
            estimator = LowPassFiniteDifferenceEstimator(alpha=params.run.estimator_lpf_alpha)

        elif variant.estimator_type == "kalman":
            Qk = np.eye(8) * 1e-6
            Rk = np.eye(4) * variant.noise_std**2
            estimator = KalmanEstimator(A, B, dt=0.001, Q=Qk, R=Rk)
    else:
        estimator = None
    
    return estimator

def build_sim_analytic(variant, params, camera_params):
    
    if params.hardware.dvs_algo is not None:
        warn("Line algorithm ignored in sim_analytic mode")
    
    vision = SimVisionModel(
                camera_params,
                noise_std=variant.noise_std,
                delay_steps=variant.delay_steps
            )
    return vision

def build_line_algo(hw):
    if hw.dvs_algo is None:
        raise ValueError("DVS modes require a line detection algorithm")
    
    elif hw.dvs_algo == "sam":
        noise_filter_ms = hw.sam_filter_ms
        
        return SamLineAlgorithm(min_points=50), SamLineAlgorithm(min_points=50), noise_filter_ms
    else:
        noise_filter_ms = None # we don't want to filter noise in hough algo
        
        return PaperHoughLineAlgorithm(params=hw.dvs_hough), PaperHoughLineAlgorithm(params=hw.dvs_hough), noise_filter_ms

def build_sim_dvs(variant, params, camera_params):
    hw = params.hardware

    cam1_algo, cam2_algo, _ = build_line_algo(hw)
    
    vision = SimEventCameraInterface(
                    camera_params=camera_params,
                    cam1_algo=cam1_algo,
                    cam2_algo=cam2_algo,
                )
    
    return vision

def connect_dvs_cameras(hw):
    from perception.dvs_camera_reader import discover_devices
    
    if hw.dvs_cam_x_port is not None and hw.dvs_cam_y_port is not None:
        return hw.dvs_cam_x_port, hw.dvs_cam_y_port
    else:
        devices = discover_devices()
        if len(devices) < 2:
            raise RuntimeError("Need at least 2 DVS cameras for real DVS mode.")
        return devices[0], devices[1]
        
def load_regression_algo(hw):
    
    if hw.dvs_use_regression:
        from perception.dvs_pose_regression_model import DVSPoseRegressionModel
        return DVSPoseRegressionModel.load("perception/calibration_files/dvs_pose_regression_model.json")
    else:
        return None

def build_real_dvs(params, camera_params):
    hw = params.hardware
    
    cam1_algo, cam2_algo, noise_filter_duration_ms = build_line_algo(hw)
    
    cam1_device, cam2_device = connect_dvs_cameras(hw)
            
    dvs_regression_model = load_regression_algo(hw)
    
    vision = RealEventCameraInterface(
                    camera_params=camera_params,
                    cam1_algo=cam1_algo,
                    cam2_algo=cam2_algo,
                    cam1_device=cam1_device,
                    cam2_device=cam2_device,
                    dvs_regression_model=dvs_regression_model,
                    noise_filter_duration_ms=noise_filter_duration_ms,
    )
    
    return vision

def build_vision(variant, params, camera_params):
    
    mode = params.hardware.vision_mode

    if mode == "real_dvs":
        return build_real_dvs(params, camera_params)

    elif mode == "sim_dvs":
        return build_sim_dvs(variant, params, camera_params)

    elif mode == "sim_analytic":
        return build_sim_analytic(variant, params, camera_params)

    else:
        return None

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

    return ServoSystem(
        mech,
        port=params.hardware.servo_port,
        frequency=params.hardware.servo_frequency,
    )

def build_visualizer(params):
    
    if not params.run.realtimerender:
        return None
    
    show_workspace = params.hardware.servo
    
    if params.hardware.vision_mode == "real_dvs":
        return DVSWorkspaceVisualizer(workspace=params.workspace, show_workspace=show_workspace)
    
    return PencilVisualizerRealtime(show_workspace=show_workspace, workspace=params.workspace)

def calculate_rates(params):
    
    actuator_rate = params.hardware.servo_frequency
    render_rate = 30
    
    actuator_dt = 1 / actuator_rate
    render_dt = 1 / render_rate
    
    return actuator_dt, render_dt

def build_scheduler(params):
    actuator_dt, render_dt = calculate_rates(params)
    return Scheduler(dt=params.run.dt, actuator_dt=actuator_dt, render_dt=render_dt)
  
def build_stop_condition(params, policy: str):
    run = params.run

    steps = int(run.total_time / run.dt)

    max_steps = MaxSteps(steps)
    fall = FallCondition()
    stabilize = StabilizedCondition(
        tol=run.stability_tolerance,
        settle_time=0.5,
    )

    if policy == StopPolicy.FIXED_TIME:
        return max_steps

    elif policy == StopPolicy.EARLY_STOP:
        return AnyStop([max_steps, fall, stabilize])

    elif policy == StopPolicy.INFINITE:
        return Infinite()

    else:
        raise ValueError(f"Unknown stop policy: {policy}")
  
def build_pacing(params):
    
    real_time = params.run.realtimerender
    
    if real_time:
        return RealTimePacing(params.run.dt)
    else:
        return NoPacing()
    

def system_factory(setup):
    
    variant = setup.default_variant
    params = setup.params
    camera_params = setup.camera_params
    
    plant = build_plant(params)

    controller = build_controller(variant, params)

    estimator = build_estimator(variant, params)
    
    vision = build_vision(variant, params, camera_params)

    perception = Perception(vision, estimator)
    
    system = System(plant, perception, controller, params.run.dt)
    
    return system

def runner_factory(params, system, stop_policy):
    
    scheduler = build_scheduler(params)
    
    stop_condition = build_stop_condition(params, stop_policy)
    
    pacing = build_pacing(params)
    
    logger = Logger()
    
    mech = build_mechanism(params)
    
    actuator = build_actuator(params, mech)
    
    visualizer = build_visualizer(params)
    
    runner = ExperimentRunner(
        system=system,
        scheduler=scheduler,
        stop_condition=stop_condition,
        pacing=pacing,
        logger=logger,
        actuator=actuator,
        visualizer=visualizer,
    )
    
    return runner
