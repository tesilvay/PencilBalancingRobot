import dv_processing as dv
import cv2 as cv
import numpy as np
import math
from collections import deque

capture = dv.io.camera.open()

if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not provide event stream.")

W, H = capture.getEventResolution()
print(f"Opened [{capture.getCameraName()}] with resolution {W}x{H}")

cv.namedWindow("DVS Angle Estimation", cv.WINDOW_NORMAL)

# ---------------------------
# Tunable Parameters
# ---------------------------

time_window_us = 5000      # 10 ms window (try 5000–20000)
min_points = 200
min_vertical_span_ratio = 0.7
angle_smoothing = 0.9

# ---------------------------
# State
# ---------------------------

event_buffer = deque()
smoothed_angle = 0.0

while capture.isRunning():

    events = capture.getNextEventBatch()
    if events is None:
        continue

    events_np = events.numpy()

    xs = events_np['x']
    ys = events_np['y']
    ts = events_np['timestamp']

    # Add new events to buffer
    for x, y, t in zip(xs, ys, ts):
        event_buffer.append((x, y, t))

    if len(event_buffer) == 0:
        continue

    current_time = event_buffer[-1][2]

    # Remove old events outside time window
    while event_buffer and (current_time - event_buffer[0][2]) > time_window_us:
        event_buffer.popleft()

    if len(event_buffer) < min_points:
        continue

    # Convert buffer to arrays
    buffer_array = np.array(event_buffer)
    xs_pts = buffer_array[:, 0].astype(np.float32)
    ys_pts = buffer_array[:, 1].astype(np.float32)

    # Vertical span filtering
    vertical_span = np.max(ys_pts) - np.min(ys_pts)
    if vertical_span < H * min_vertical_span_ratio:
        continue

    print("Points used:", len(xs_pts))
    print("Vertical span:", vertical_span)

    coords = np.column_stack((xs_pts, ys_pts))
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid

    cov = np.dot(centered.T, centered) / len(centered)
    eigvals, eigvecs = np.linalg.eig(cov)

    direction = eigvecs[:, np.argmax(eigvals)]
    dx, dy = direction

    # Ensure consistent orientation
    if dy < 0:
        dx = -dx
        dy = -dy

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    smoothed_angle = (
        angle_smoothing * smoothed_angle +
        (1 - angle_smoothing) * angle_deg
    )

    print("Angle:", smoothed_angle)
    print("-------------")

    # Visualization
    display = np.zeros((H, W, 3), dtype=np.uint8)

    # Draw current window events
    display[ys_pts.astype(int), xs_pts.astype(int)] = (255, 255, 255)

    # Draw fitted line
    length = H

    x0 = int(centroid[0] - dx * length)
    y0 = int(centroid[1] - dy * length)

    x1 = int(centroid[0] + dx * length)
    y1 = int(centroid[1] + dy * length)

    cv.line(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv.imshow("DVS Angle Estimation", display)
    cv.waitKey(1)