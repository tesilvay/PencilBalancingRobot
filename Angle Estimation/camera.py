import numpy as np


class AverageEvents:
    """
    Event-based line estimator for a DVS camera.

    Maintains an activity surface and performs regression
    to estimate the line

        x = s*y + b

    representing the pencil projection in the image.

    update(events_np) returns:
        s : slope
        b : intercept
    """

    def __init__(self, width, height, decay=0.8, threshold=0.5, min_points=50):
        self.W = width
        self.H = height

        self.decay = decay
        self.threshold = threshold
        self.min_points = min_points

        self.surface = np.zeros((height, width), dtype=np.float32)

        self.s = None
        self.b = None

    def update(self, events_np):
        """
        Update estimator with a batch of events.

        events_np must contain fields 'x' and 'y'.

        Returns:
            (s, b) slope and intercept of the detected line
            or (None, None) if insufficient data.
        """

        xs = events_np['x']
        ys = events_np['y']

        # Decay previous activity
        self.surface *= self.decay

        # Add new events
        self.surface[ys, xs] += 1.0

        # Threshold active pixels
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

        denom = (N * S_yy - S_y * S_y)

        if abs(denom) < 1e-6:
            return None, None

        s = (N * S_xy - S_y * S_x) / denom
        b = (S_x - s * S_y) / N

        self.s = s
        self.b = b

        return s, b
    

capture = dv.io.camera.open()
W, H = capture.getEventResolution()

estimator = AverageEvents(W, H)

while capture.isRunning():

    events = capture.getNextEventBatch()
    if events is None:
        continue

    events_np = events.numpy()

    s, b = estimator.update(events_np)

    if s is not None:
        print(f"slope={s:.4f}  intercept={b:.2f}")