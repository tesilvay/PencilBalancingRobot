import numpy as np

from core.sim_types import CameraObservation


class DVSLineAlgorithm:
    """Base class for DVS line estimation algorithms."""

    def update(self, events_np):
        raise NotImplementedError

    def reset(self):
        pass


class SurfaceRegressionAlgorithm(DVSLineAlgorithm):
    """
    Activity surface + linear regression.
    """

    def __init__(self, width, height, decay=0.8, threshold=0.5, min_points=50):

        self.W = width
        self.H = height

        self.decay = decay
        self.threshold = threshold
        self.min_points = min_points

        self.surface = np.zeros((height, width), dtype=np.float32)

    def update(self, events_np):

        xs = events_np['x']
        ys = events_np['y']

        # decay previous activity
        self.surface *= self.decay

        # accumulate events
        self.surface[ys, xs] += 1.0

        mask = self.surface > self.threshold
        points = np.column_stack(np.where(mask))  # (y,x)

        if len(points) < self.min_points:
            return None, None

        ys_pts = points[:, 0].astype(np.float32)
        xs_pts = points[:, 1].astype(np.float32)

        N = len(xs_pts)

        S_y = np.sum(ys_pts)
        S_yy = np.sum(ys_pts * ys_pts)
        S_x = np.sum(xs_pts)
        S_xy = np.sum(xs_pts * ys_pts)

        denom = N * S_yy - S_y * S_y

        if abs(denom) < 1e-6:
            return None, None

        s = (N * S_xy - S_y * S_x) / denom
        b = (S_x - s * S_y) / N

        return s, b


class PaperHoughLineAlgorithm(DVSLineAlgorithm):
    """
    Event-driven quadratic Hough tracker from the pencil balancing paper.
    """

    def __init__(self, width=346, height=260, decay=0.999, min_q=1e-6):

        self.cx = width / 2
        self.cy = height / 2

        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.E = 0.0

        self.decay = decay
        self.min_q = min_q

    def update(self, events_np):

        # center coordinates
        xs = events_np["x"] - self.cx
        ys = events_np["y"] - self.cy

        # decay ONCE per batch
        self.A *= self.decay
        self.B *= self.decay
        self.C *= self.decay
        self.D *= self.decay
        self.E *= self.decay

        for x, y in zip(xs, ys):

            self.A += y * y
            self.B += 2.0 * y
            self.C += 1.0
            self.D += -2.0 * x * y
            self.E += -2.0 * x

        q = 4 * self.A * self.C - self.B * self.B
        #print(f"Q: {q}")

        if abs(q) < self.min_q:
            return None, None

        b_center = (self.D * self.B - 2 * self.A * self.E) / q
        m = (self.B * self.E - 2 * self.C * self.D) / q

        # convert centered intercept back to pixel intercept
        b_pixel = b_center + self.cx - m * self.cy
        
        
        return CameraObservation(
            slope=m,
            intercept=b_pixel
        )

    def reset(self):
        self.A = self.B = self.C = self.D = self.E = 0.0

