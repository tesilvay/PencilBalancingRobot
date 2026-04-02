from new_architecture.spec import Spec

PLANT_REGISTRY = {
    "sim": Spec(BalancerPlant, PlantParams, PLANT_PRESETS),
    "null": Spec(NullPlant, NullParams, NULL_PRESETS),
}

CONTROLLER_REGISTRY = {
    "pole": Spec(PoleController, PoleParams, POLE_PRESETS, registries={"plant": PLANT_REGISTRY}),
    "lqr": Spec(LQRController, LQRParams, LQR_PRESETS, registries={"plant": PLANT_REGISTRY}),
    "smooth_pole": Spec(SmoothPoleController, SmoothPoleParams, SMOOTH_POLE_PRESETS, registries={"plant": PLANT_REGISTRY}),
    "circle": Spec(CircleController, CircleParams, CIRCLE_PRESETS, registries={"plant": PLANT_REGISTRY}),
    "null": Spec(NullController, NullParams, NULL_PRESETS),
}

from system.estimator import ESTIMATOR_REGISTRY  # noqa: E402

LINE_ALGO_REGISTRY = {
    "hough": Spec(PaperHoughLineAlgorithm, HoughLineParams, HOUGH_PRESETS),
    "sam": Spec(SamLineAlgorithm, SamLineParams, SAM_PRESETS),
}

REG_MODEL_REGISTRY = {
    "none": Spec(NullRegression, NullParams, NULL_PRESETS),
    "simple": Spec(SimpleDVSRegressionModel, SimpleRegressionParams, SIMPLE_REG_PRESETS),
}

VISION_INTERFACE_REGISTRY = {
    "sim_analytic": Spec(SimVisionModel, SimAnalyticParams, SIM_ANALYTIC_PRESETS, sim_only=True),
    "sim_dvs": Spec(SimEventCameraInterface, SimDVSParams, SIM_DVS_PRESETS, sim_only=True),
    "real_dvs": Spec(RealEventCameraInterface, RealDVSParams, REAL_DVS_PRESETS, sim_only=False),
}

VISION_REGISTRY = {
    "default": Spec(
        cls       = Vision,
        Params    = VisionParams,
        Presets   = VISION_PRESETS,
        registries={
            "interface": VISION_INTERFACE_REGISTRY,
            "algo":      LINE_ALGO_REGISTRY,
            "reg_model": REG_MODEL_REGISTRY,
        },
    )
}

ACTUATOR_REGISTRY = {
    "servo": Spec(ServoController, ServoParams, SERVO_PRESETS, sim_only=False),
    "mock": Spec(MockServoController, NullParams, NULL_PRESETS, sim_only=True),
}

SUPERVISOR_REGISTRY = {
    "dynamic": Spec(DynamicSupervisor, DynamicSupervisorParams, DYNAMIC_SUPERVISOR_PRESETS), # multi states
    "static": Spec(StaticSupervisor, StaticSupervisorParams, STATIC_SUPERVISOR_PRESETS),  # single state
}

SYSTEM_REGISTRY = {
    "default": Spec(
        cls      = System,
        Params   = SystemParams,
        Presets  = SYSTEM_PRESETS,
        registries = {
            "plant":       PLANT_REGISTRY,
            "controllers": CONTROLLER_REGISTRY,  # dict of objects
            "estimators":  ESTIMATOR_REGISTRY,   # dict of objects
            "vision":      VISION_REGISTRY,       # composite; interface string + nested algo / reg_model
            "actuator":    ACTUATOR_REGISTRY,
            "supervisor":  SUPERVISOR_REGISTRY,
        }
    )
}

LOGGER_REGISTRY = {
    "default": Spec(Logger, NullParams, NULL_PRESETS),
}

STOP_CONDITION_REGISTRY = {
    "fall": Spec(FallCondition, FallConditionParams, FALL_CONDITION_PRESETS),
    "stabilized": Spec(StabilizedCondition, StabilizedParams, STABILIZED_CONDITION_PRESETS),
    "max_steps": Spec(MaxStepsCondition, MaxStepsConditionParams, MAX_STEPS_CONDITION_PRESETS),
    "any": Spec(AnyStopCondition, AnyStopConditionParams, ANY_STOP_CONDITION_PRESETS),
    "infinite": Spec(InfiniteCondition, NullParams, NULL_PRESETS),
}

VISUALIZER_REGISTRY = {
    # base realtime visualizers
    "sim": Spec(SimDvsVisualizer, SimDvsVisualizerParams, SIM_DVS_VISUALIZER_PRESETS),
    "real": Spec(RealDvsVisualizer, RealDvsVisualizerParams, REAL_DVS_VISUALIZER_PRESETS),
    "one": Spec(OneDvsVisualizer, OneDvsVisualizerParams, ONE_DVS_VISUALIZER_PRESETS),

    # workspace variants
    "sim_ws": Spec(SimDvsWorkspaceVisualizer, SimDvsWorkspaceVisualizerParams, SIM_DVS_WORKSPACE_VISUALIZER_PRESETS),
    "real_ws": Spec(RealDvsWorkspaceVisualizer, RealDvsWorkspaceVisualizerParams, REAL_DVS_WORKSPACE_VISUALIZER_PRESETS),
    
    # 3d animation
    "3d": Spec(Visualizer3D, Visualizer3DParams, VISUALIZER_3D_PRESETS),
}

PROGRESS_REGISTRY = {
    "default": Spec(ConsoleProgress, ProgressParams, PROGRESS_PRESETS),
}

PACING_REGISTRY = {
    "realtime": Spec(RealTimePacing, RealTimePacingParams, REALTIME_PACING_PRESETS),
    "null":  Spec(NoPacing, NullParams, NULL_PRESETS),
}

SCHEDULER_REGISTRY = {
    "realtime": Spec(Scheduler, SchedulerParams, SCHEDULER_PRESETS),
}

EXPERIMENT_REGISTRY = {
    "default": Spec(
        cls      = Experiment,
        Params   = ExperimentParams,
        Presets  = EXPERIMENT_PRESETS,
        registries = {
            "system":          SYSTEM_REGISTRY,
            "logger":          LOGGER_REGISTRY,
            "stop_condition":  STOP_CONDITION_REGISTRY,
            "visualizer":      VISUALIZER_REGISTRY,
            "progress":        PROGRESS_REGISTRY,
            "pacing":          PACING_REGISTRY,
            "scheduler":       SCHEDULER_REGISTRY,
        }
    )
}
