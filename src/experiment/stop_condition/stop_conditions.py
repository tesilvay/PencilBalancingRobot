from numpy import deg2rad

class StopCondition:
    def reset(self):
        pass

    def should_stop(self, i, state, dt):
        raise NotImplementedError

    def is_stabilized(self):
        return False
    
    def settling_time(self):
        return None

class FallCondition(StopCondition):
    def __init__(self, max_angle=deg2rad(45)):
        self.max_angle = max_angle

    def should_stop(self, i, state, dt):
        return (
            abs(state.ax) > self.max_angle
            or abs(state.ay) > self.max_angle
        )

class StabilizedCondition(StopCondition):
    def __init__(self, tol_ang, tol_m, settle_time):
        self.tol_ang = tol_ang
        self.tol_m = tol_m
        self.settle_time = settle_time
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None

    def reset(self):
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None
    
    def _is_inside_tolerance(self, state):
        return (
            abs(state.ax) < self.tol_ang
            and abs(state.ay) < self.tol_ang
            and abs(state.px) < self.tol_m
            and abs(state.py) < self.tol_m
        )

    def should_stop(self, i, state, dt):
        if (self._is_inside_tolerance(state)):
            self.time_in_tol += dt
        else:
            self.time_in_tol = 0.0

        if (not self._stabilized) and self.time_in_tol >= self.settle_time:
            self._stabilized = True
            self._settling_time = i * dt
            return True  # only matters in batch mode

        return False

    def is_stabilized(self):
        return self._stabilized
    
    def settling_time(self):
        return self._settling_time

class MaxSteps(StabilizedCondition):
    def __init__(self, steps, tol_ang, tol_m, settle_time):
        super().__init__(tol_ang=tol_ang, tol_m=tol_m, settle_time=settle_time)
        self.steps = steps

    def should_stop(self, i, state, dt):
        # Run stabilization logic, but ignore its stop signal
        super().should_stop(i, state, dt)

        # Only stop based on step count
        return i >= self.steps

class AnyStop(StopCondition):
    def __init__(self, conditions):
        self.conditions = conditions

    def reset(self):
        for c in self.conditions:
            if hasattr(c, "reset"):
                c.reset()

    def should_stop(self, i, state, dt):
        return any(c.should_stop(i, state, dt) for c in self.conditions)
    
    def is_stabilized(self):
        return any(
            getattr(c, "is_stabilized", lambda: False)()
            for c in self.conditions
        )
    
    def settling_time(self):
        for c in self.conditions:
            if hasattr(c, "settling_time"):
                t = c.settling_time()
                if t is not None:
                    return t
        return None

class Infinite(StopCondition):
    def should_stop(self, i, state, dt):
        return False
