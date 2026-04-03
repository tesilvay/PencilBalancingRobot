from dataclasses import dataclass


@dataclass
class SystemParams:
    plant:       object
    controllers: dict
    estimators:  dict
    sensor:      object
    actuator:    object
    supervisor:  object


SYSTEM_PRESETS = {
    "dynamic_sim": {
        "plant":       "sim:default",
        "controllers": {"follower": "pole:default", "smooth": "smooth_pole:default"},
        "estimators":  {"lpf": "lpf:default", "kalman": "kalman:default"},
        "sensor":      "sim_dvs:hough",
        "actuator":    "mock:default",
        "supervisor":  "dynamic:default",
    },
    "simple_sim": {
        "base": "dynamic_sim",
        "controllers": {"smooth": "smooth_pole:default"},
        "estimators":  {"kalman": "kalman:default"},
        "supervisor":  "static:default",
    },
    "real": {
        "base": "dynamic_sim",
        "sensor":   "real_dvs:hough",
        "actuator": "servo:default",
    },
}


class System:
    def __init__(self, params: SystemParams):
        self.plant       = params.plant
        self.controllers = params.controllers   # dict[str, Controller]
        self.estimators  = params.estimators    # dict[str, Estimator]
        self.sensor      = params.sensor
        self.actuator    = params.actuator
        self.supervisor  = params.supervisor

        self.active_controller = self.controllers.get(
            "follower", next(iter(self.controllers.values()))
        )
        self.active_estimator = self.estimators.get(
            "lpf", next(iter(self.estimators.values()))
        )
        self.state = None
        self.u     = None

    def step(self, dt):
        # 1. estimate and control with current actives
        x_est, innovation = self.active_estimator.estimate(self.sensor.read()) # estimator calculates innovation too
        u                 = self.active_controller.compute(x_est)
        
        self.actuator.apply(u)

        # 2. supervisor decides what should be active next step
        ctrl_key, est_key = self.supervisor.update(x_est, innovation, dt)

        # 3. system owns the swap — including warm-start on estimator switch
        new_estimator = self.estimators[est_key]
        if new_estimator is not self.active_estimator:
            new_estimator.initialize_from(self.active_estimator)

        self.active_controller = self.controllers[ctrl_key]
        self.active_estimator  = new_estimator
        self.state = x_est
        self.u     = u