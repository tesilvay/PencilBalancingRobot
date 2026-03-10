import dv_processing as dv
import cv2 as cv
import numpy as np
import math

capture = dv.io.camera.open()

if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not provide event stream.")

W, H = capture.getEventResolution()
print(f"Opened [{capture.getCameraName()}] with resolution {W}x{H}")

cv.namedWindow("DVS Angle Estimation", cv.WINDOW_NORMAL)

surface = np.zeros((H, W), dtype=np.float32)

decay = 0.8
threshold = 0.5  # activity threshold for regression

while capture.isRunning():

    events = capture.getNextEventBatch()
    if events is None:
        continue

    events_np = events.numpy()
    xs = events_np['x']
    ys = events_np['y']

    # Decay old surface
    surface *= decay

    # Accumulate new events (ignore polarity for geometry)
    surface[ys, xs] += 1.0

    # Copy for visualization
    display = surface.copy()
    display = np.clip(display, 0, 5)
    display = (display / 5.0 * 255).astype(np.uint8)

    # --- LINE FITTING ---
    mask = surface > threshold
    points = np.column_stack(np.where(mask))  # (y, x)

    if len(points) > 50:  # need enough points

        ys_pts = points[:, 0].astype(np.float32)
        xs_pts = points[:, 1].astype(np.float32)

        N = len(xs_pts)
        S_y = np.sum(ys_pts)
        S_yy = np.sum(ys_pts * ys_pts)
        S_x = np.sum(xs_pts)
        S_xy = np.sum(xs_pts * ys_pts)

        denom = (N * S_yy - S_y * S_y)

        if abs(denom) > 1e-6:
            b = (N * S_xy - S_y * S_x) / denom
            a = (S_x - b * S_y) / N

            # Compute angle (radians → degrees)
            angle_rad = math.atan(b)
            angle_deg = math.degrees(angle_rad)

            print(f"Angle: {angle_deg:.2f} degrees")

            # Draw line
            y0 = 0
            y1 = H - 1
            x0 = int(a + b * y0)
            x1 = int(a + b * y1)

            cv.line(display, (x0, y0), (x1, y1), 255, 2)

    cv.imshow("DVS Angle Estimation", display)
    cv.waitKey(1)