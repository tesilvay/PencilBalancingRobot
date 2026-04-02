class StopCondition:
    def reset(self):
        pass

    def should_stop(self, i, state, dt):
        raise NotImplementedError

    def is_stabilized(self):
        return False
    
    def settling_time(self):
        return None
