from dataclasses import dataclass, field
from src.shared import NullParams


@dataclass
class SystemParams:
    plant:       object
    controllers: list
    estimators:  list
    sensor:      object
    actuator:    object
    supervisor:  object


SYSTEM_PRESETS = {
    "dynamic_sim": {
        "plant":       "sim:default",
        "controllers": ["follower:default","smooth_pole:default"],
        "estimators":  ["lpf:default", "kalman:default"],
        "sensor":      "sim_dvs:hough",
        "actuator":    "mock:default",
        "supervisor":  "dynamic:default",
    },
    "simple_sim": {
        "base": "dynamic_sim",
        "controllers": ["smooth_pole:default"],
        "estimators":  ["kalman:default"],
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
        self.x = None
        self.u = None

    def step(self, dt):
        
        x_true, acc = self.plant.step(self.x, self.u, dt)
        
        # get measurements
        y = self.sensor.get_y(x_true)
        
        # every estimator calculates innovation too
        x_hat, innovation = self.active_estimator.estimate(y_meas=y, dt=dt, u_cmd=self.u) 
        u_cmd = self.active_controller.compute(x_hat)
        
        self.actuator.apply(u_cmd)

        # 2. supervisor decides what should be active next step
        ctrl_key, est_key = self.supervisor.update(x_hat, innovation, dt)

        # 3. system owns the swap — including warm-start on estimator switch
        new_estimator = self.estimators[est_key]
        if new_estimator is not self.active_estimator:
            new_estimator.reset(x_hat)

        self.active_controller = self.controllers[ctrl_key]
        self.active_estimator  = new_estimator
        self.x = x_hat
        self.u = u_cmd