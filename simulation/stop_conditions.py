

class StopCondition:
    def reset(self):
        pass

    def should_stop(self, i, state, dt):
        raise NotImplementedError

    def is_stabilized(self):
        return False
    
    def settling_time(self):
        return None

class MaxSteps(StopCondition):
    def __init__(self, steps):
        self.steps = steps

    def should_stop(self, i, state, dt):
        return i >= self.steps

class FallCondition(StopCondition):
    def __init__(self, max_angle=2.5):
        self.max_angle = max_angle

    def should_stop(self, i, state, dt):
        return (
            abs(state.alpha_x) > self.max_angle
            or abs(state.alpha_y) > self.max_angle
        )

class StabilizedCondition(StopCondition):
    def __init__(self, tol, settle_time):
        self.tol = tol
        self.settle_time = settle_time
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None

    def reset(self):
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None

    def should_stop(self, i, state, dt):
        if (
            abs(state.alpha_x) < self.tol
            and abs(state.alpha_y) < self.tol
        ):
            self.time_in_tol += dt
        else:
            self.time_in_tol = 0.0

        if self.time_in_tol >= self.settle_time:
            self._stabilized = True
            self._settling_time = i * dt
            return True  # only matters in batch mode

        return False

    def is_stabilized(self):
        return self._stabilized
    
    def settling_time(self):
        return self._settling_time
    
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
