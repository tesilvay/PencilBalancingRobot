import dv_processing as dv
import cv2 as cv
import numpy as np
import math

capture = dv.io.camera.open()

if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not provide event stream.")

W, H = capture.getEventResolution()
print(f"Opened [{capture.getCameraName()}] with resolution {W}x{H}")

cv.namedWindow("Fast jAER-style Tracker", cv.WINDOW_NORMAL)

# --------------------------
# Tunable Parameters
# --------------------------

sigma = 6.0           # gating width (3–10)
alpha = 0.00005       # learning rate (tune carefully)
min_events = 50       # require some events per batch

# --------------------------
# Persistent Line Model
# --------------------------

a = W / 2
b = 0.0

while capture.isRunning():

    events = capture.getNextEventBatch()
    if events is None:
        continue

    events_np = events.numpy()
    xs = events_np['x']
    ys = events_np['y']

    if len(xs) < min_events:
        continue

    # Normalize y for stability
    y_norm = ys / H

    for x_e, y_e, y_n in zip(xs, ys, y_norm):

        x_pred = a + b * y_e
        error = x_e - x_pred

        w = math.exp(-(error * error) / (2 * sigma * sigma))

        # Incremental gradient update
        a += alpha * w * error
        b += alpha * w * error * y_n

    angle_rad = math.atan(b)
    angle_deg = math.degrees(angle_rad)

    print(f"Angle: {angle_deg:.2f}")

    # --------------------------
    # Visualization
    # --------------------------

    display = np.zeros((H, W, 3), dtype=np.uint8)
    display[ys.astype(int), xs.astype(int)] = (255, 255, 255)

    y0 = 0
    y1 = H - 1
    x0 = int(a + b * y0)
    x1 = int(a + b * y1)

    cv.line(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv.imshow("Fast jAER-style Tracker", display)
    cv.waitKey(1)