import time

class Pacing:
    def pace(self): ...

class NoPacing(Pacing):
    def pace(self):
        pass

class RealTimePacing(Pacing):
    def __init__(self, dt):
        self.dt = dt
        self.next_time = time.perf_counter()

    def pace(self):
        self.next_time += self.dt
        sleep = self.next_time - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
    