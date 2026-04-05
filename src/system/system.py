from dataclasses import dataclass, field
import numpy as np
from src.shared import (
    NullParams,
    State,
    InitConditionsSpread,
    default_spread,
    ControlInput,
    StepData,
    TableAccel,
)

@dataclass
class SystemParams:
    plant:       object
    # Ordered lists of collaborators sharing a common base type. Supervisors
    # select active instances by integer index; order must match preset + supervisor.
    controllers: list
    estimators:  list
    sensor:      object
    actuator:    object
    supervisor:  object
    init_spread: InitConditionsSpread = field(default_factory=default_spread)


SYSTEM_PRESETS = {
    "dynamic_sim": {
        "plant":       "sim:default",
        "controllers": ["pole:default", "smooth_pole:default"],
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
        "sensor":      "sim_analytic:default",
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
        self.controllers = params.controllers
        self.estimators  = params.estimators
        self.sensor      = params.sensor
        self.actuator    = params.actuator
        self.supervisor  = params.supervisor
        self.init_spread = params.init_spread

        if not self.controllers or not self.estimators:
            raise ValueError("System requires non-empty controllers and estimators lists.")
        self.active_controller = self.controllers[0]
        self.active_estimator  = self.estimators[0]
        
        self.step_data: StepData | None = None
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
        ctrl_i, est_i = self.supervisor.update(x_hat, innovation, dt)

        # 3. system owns the swap — including warm-start on estimator switch
        new_estimator = self.estimators[est_i]
        if new_estimator is not self.active_estimator:
            new_estimator.reset(x_hat)

        self.active_controller = self.controllers[ctrl_i]
        self.active_estimator = new_estimator
        
        self.x = x_hat
        self.u = u_cmd
        
        self.step_data = StepData(x=x_hat, u=u_cmd, acc=acc, innovation=innovation)
    
    def reset(self):
        self.active_controller.reset()
        self.active_estimator.reset()
        self.x = self.random_state()
        self.u = ControlInput(px_cmd=0, py_cmd=0)
        
        self.step_data = StepData(
            x=self.x,
            u=self.u,
            acc=TableAccel(x_ddot=0.0, y_ddot=0.0),
            innovation=np.zeros(4),
        )
    
    
    # doesn't take x_ref into account yet
    def random_state(self) -> State:
        spread  = lambda r: np.random.uniform(-r, r)
        
        angle = np.deg2rad(self.init_spread.ang_deg)
        pos = self.init_spread.pos_m
        v_spread = self.init_spread.vel_mps
        w_spread = np.deg2rad(self.init_spread.w_degps)

        return State(
            px = spread(pos),
            vx = spread(v_spread),
            ax = spread(angle),
            wx = spread(w_spread),

            py = spread(pos),
            vy = spread(v_spread),
            ay = spread(angle),
            wy = spread(w_spread),
        )