import numpy as np
from core.system_builder import system_factory, runner_factory

from core.sim_types import (
    ExperimentSetup,
    SystemState,
    TableCommand,
    StopPolicy,
)

from hardware.servo_workspace_offset_calibrator import calibrate_servo_workspace_offset


class SimulationEngine:
    def __init__(self, stop_policy: StopPolicy):
        self.stop_policy = stop_policy
    
    def run(self, setup: ExperimentSetup):
        system = system_factory(setup)
        runner = runner_factory(setup.params, system, self.stop_policy)

        if self._should_calibrate_servo_offset(setup.params):
            if runner.actuator is None:
                raise RuntimeError("Servo offset calibration requested but runner has no actuator.")

            x_offset, y_offset = calibrate_servo_workspace_offset(
                system=system,
                actuator=runner.actuator,
                workspace=setup.params.workspace,
            )
            runner.actuator.set_workspace_offset(x_offset, y_offset)

            # Best-effort state reset so the first run doesn't start with stale pose history.
            try:
                vision = getattr(system.perception, "vision", None)
                if vision is not None and hasattr(vision, "reset"):
                    vision.reset()
            except Exception:
                pass
            try:
                estimator = getattr(system.perception, "estimator", None)
                if estimator is not None and hasattr(estimator, "reset"):
                    estimator.reset()
            except Exception:
                pass
            try:
                if hasattr(system.perception, "state_est"):
                    system.perception.state_est = None
            except Exception:
                pass

        initial_state, initial_command = self._initialize(setup)

        runner.initialize(initial_state, initial_command)
        return runner.run()

    def _initialize(self, setup):
        params = setup.params
        ws = params.workspace

        x_ref = SystemState(
            x=ws.x_ref, x_dot=0.0, alpha_x=0.0, alpha_x_dot=0.0,
            y=ws.y_ref, y_dot=0.0, alpha_y=0.0, alpha_y_dot=0.0
        )

        angle = np.deg2rad(params.run.initial_angle_spread_deg)
        spread = params.run.initial_position_spread_m

        state = SystemState(
            x=x_ref.x + np.random.uniform(-spread, spread),
            x_dot=0.0,
            alpha_x=np.random.uniform(-angle, angle),
            alpha_x_dot=0.0,
            y=x_ref.y + np.random.uniform(-spread, spread),
            y_dot=0.0,
            alpha_y=np.random.uniform(-angle, angle),
            alpha_y_dot=0.0,
        )

        cmd = TableCommand(x_ref.x, x_ref.y)

        return state, cmd

    @staticmethod
    def _should_calibrate_servo_offset(params) -> bool:
        """
        Calibrate the real mechanism offset only when:
        - real servos are enabled (servo_port is non-None)
        Calibration is intentionally run on every connection to a real servo port,
        regardless of vision mode. Pose readouts are best-effort.
        """
        hw = getattr(params, "hardware", None)
        if hw is None:
            return False
        if not getattr(hw, "servo", False):
            return False
        if getattr(hw, "servo_port", None) is None:
            return False
        return True


class RealTimeEngine(SimulationEngine):
    pass  # same execution, different runner internally if needed
