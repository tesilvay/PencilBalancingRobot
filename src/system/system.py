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
    WorkspaceParams,
    clamp_control_input_to_workspace,
    default_workspace,
)

@dataclass
class SystemParams:
    plants:       list
    # Ordered lists of collaborators sharing a common base type. Supervisors
    # select active instances by integer index; order must match preset + supervisor.
    controllers: list
    estimators:  list
    sensor:      object
    actuator:    object
    supervisor:  object
    init_spread: InitConditionsSpread = field(default_factory=default_spread)
    workspace:   WorkspaceParams     = field(default_factory=default_workspace)


SYSTEM_PRESETS = {
    "simple_sim": {
        "plants":       ["sim:default"],
        "controllers": ["smooth_pole:default"],
        "estimators":  ["lpf:default"],
        "sensor":      "sim_analytic:default",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
    },
    "placing_only": {
        "plants":       ["placing:steady_hands"],
        "controllers": ["null:default"],
        "estimators":  ["lpf:default"],
        "sensor":      "sim_analytic:default",
        "actuator":    "mock:default",
        "supervisor":  "static:default",
    },
    "dynamic_sim": {
        "plants":       ["placing:steady_hands", "sim:default"],
        "controllers": ["null:default", "smooth_pole:default"],
        "estimators":  ["lpf:default", "kalman:default"],
        "sensor":      "sim_dvs:hough",
        "actuator":    "mock:default",
        "supervisor":  "dynamic:default",
    },
    "real": {
        "base": "dynamic_sim",
        "sensor":   "real_dvs:hough",
        "actuator": "servo:default",
    },
}



class System:
    def __init__(self, params: SystemParams):
        self.plants      = params.plants
        self.controllers = params.controllers
        self.estimators  = params.estimators
        self.sensor      = params.sensor
        self.actuator    = params.actuator
        self.supervisor  = params.supervisor
        self.init_spread = params.init_spread
        self.workspace   = params.workspace
        
        self.active_plant  = self.plants[0]
        self.active_controller = self.controllers[0]
        self.active_estimator  = self.estimators[0]
        
        
        self.step_data: StepData | None = None
        self.x = None
        self.u = None
        self.last_y_meas = None

    def finalize_command(self, u_raw: ControlInput) -> ControlInput:
        u_applied = clamp_control_input_to_workspace(u_raw, self.workspace)
        self.active_controller.set_applied_command(u_applied)
        return u_applied

    def step(self, dt):
        
        x_true, acc = self.active_plant.step(self.x, self.u, dt)
        
        # get measurements
        y = self.sensor.get_y(x_true)
        self.last_y_meas = y
        
        # every estimator calculates innovation too
        x_hat, innovation = self.active_estimator.estimate(y_meas=y, dt=dt, u_cmd=self.u) 
        u_raw = self.active_controller.compute(x_hat)
        u_cmd = self.finalize_command(u_raw)

        mech_joints = self.actuator.apply(u_cmd)

        # 2. supervisor decides what should be active next step
        ctrl_i, est_i = self.supervisor.update(x_hat, innovation, dt)

        # 3. system owns the swap — including warm-start on estimator switch
        new_estimator = self.estimators[est_i]
        if new_estimator is not self.active_estimator:
            new_estimator.reset(x_hat)

        # plant and controller change equally
        self.active_plant = self.plants[ctrl_i]
        self.active_controller = self.controllers[ctrl_i]
        self.active_estimator = new_estimator
        
        self.x = x_true
        self.u = u_cmd

        self.step_data = StepData(
            x=x_true,
            u=u_cmd,
            acc=acc,
            innovation=innovation,
            mech_joints=mech_joints,
        )
    
    def reset(self):
        self.active_controller.reset()
        self.active_estimator.reset()
        if hasattr(self.active_plant, "reset"):
            self.active_plant.reset()
        if hasattr(self.sensor, "reset"):
            self.sensor.reset()
        self.x = self.random_state()
        self.u = ControlInput(px_cmd=0, py_cmd=0)
        self.last_y_meas = None

        mj0 = self.actuator.mech_joint_snapshot(self.u)
        self.step_data = StepData(
            x=self.x,
            u=self.u,
            acc=TableAccel(x_ddot=0.0, y_ddot=0.0),
            innovation=np.zeros(4),
            mech_joints=mj0,
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