import dv_processing as dv
import cv2 as cv
import numpy as np
import math
import time
import datetime

devices = dv.io.camera.discover()
print(f"Discovered devices: {devices}")

capture1 = dv.io.camera.open(devices[0])
capture2 = dv.io.camera.open(devices[1])

if not capture1.isEventStreamAvailable():
    raise RuntimeError("Camera 1 does not provide event stream.")

if not capture2.isEventStreamAvailable():
    raise RuntimeError("Camera 2 does not provide event stream.")

W1, H1 = capture1.getEventResolution()
print(f"Opened [{capture1.getCameraName()}] with resolution {W1}x{H1}")

W2, H2 = capture2.getEventResolution()
print(f"Opened [{capture2.getCameraName()}] with resolution {W2}x{H2}")

cv.namedWindow("DVS Angle Estimation xy plane", cv.WINDOW_NORMAL)

cv.namedWindow("DVS Angle Estimation zy plane", cv.WINDOW_NORMAL)

surface1 = np.zeros((H1, W1), dtype=np.float32)
surface2 = np.zeros((H2, W2), dtype=np.float32)

decay = 0.5
decay_interval = 5
frame_counter = 0

threshold = 2

# Display timing (limit GUI to ~30 FPS)
last_display = time.time()
display_period = 0.03

x0_1 = x1_1 = y0_1 = y1_1 = None
x0_2 = x1_2 = y0_2 = y1_2 = None

kernel = np.ones((3,3), dtype=np.uint8)

cameraFilter1 = dv.noise.BackgroundActivityNoiseFilter((W1, H1), backgroundActivityDuration = datetime.timedelta(microseconds=30000))
cameraFilter2 = dv.noise.BackgroundActivityNoiseFilter((W2, H2), backgroundActivityDuration = datetime.timedelta(microseconds=30000))

while capture1.isRunning() and capture2.isRunning():

    events1 = capture1.getNextEventBatch()
    events2 = capture2.getNextEventBatch()

    if events1 is None and events2 is None:
        continue

    if events1 is not None:
        cameraFilter1.accept(events1)
        events1f = cameraFilter1.generateEvents()
        events1_np = events1f.numpy()
        xs1 = events1_np['x']
        ys1 = events1_np['y']
        surface1[ys1, xs1] += 1.0
        mask1 = surface1 > threshold
        mask1 = cv.morphologyEx(mask1.astype(np.uint8), cv.MORPH_OPEN, kernel)
        mask1 = mask1.astype(bool)
        ys1_pts, xs1_pts = np.where(mask1)

    if events2 is not None:
        cameraFilter2.accept(events2)
        events2f = cameraFilter2.generateEvents()
        events2_np = events2f.numpy()
        xs2 = events2_np['x']
        ys2 = events2_np['y']
        surface2[ys2, xs2] += 1.0
        mask2 = surface2 > threshold
        mask2 = cv.morphologyEx(mask2.astype(np.uint8), cv.MORPH_OPEN, kernel)
        mask2 = mask2.astype(bool)
        ys2_pts, xs2_pts = np.where(mask1)

    frame_counter += 1

    if frame_counter % decay_interval == 0:
        surface1 *= decay
        surface2 *= decay

    # --- LINE FITTING using event coordinates directly ---
    if len(xs1) > 50:

        xs1_pts = xs1.astype(np.float32)
        ys1_pts = ys1.astype(np.float32)

        N1 = len(xs1_pts)

        S_y1 = np.sum(ys1_pts)
        S_yy1 = np.sum(ys1_pts * ys1_pts)
        S_x1 = np.sum(xs1_pts)
        S_xy1 = np.sum(xs1_pts * ys1_pts)

        denom1 = (N1 * S_yy1 - S_y1 * S_y1)

        if abs(denom1) > 1e-6:

            b1 = (N1 * S_xy1 - S_y1 * S_x1) / denom1
            a1 = (S_x1 - b1 * S_y1) / N1

            angle1_rad = math.atan(b1)
            angle1_deg = math.degrees(angle1_rad)

            y0_1 = 0
            y1_1 = H1 - 1
            x0_1 = int(a1 + b1 * y0_1)
            x1_1 = int(a1 + b1 * y1_1)

            print(f"xy plane: Angle: {angle1_deg:.2f} degrees | x-intercept: {x0_1}")

    if len(xs2) > 50:

        xs2_pts = xs2.astype(np.float32)
        ys2_pts = ys2.astype(np.float32)

        N2 = len(xs2_pts)

        S_y2 = np.sum(ys2_pts)
        S_yy2 = np.sum(ys2_pts * ys2_pts)
        S_x2 = np.sum(xs2_pts)
        S_xy2 = np.sum(xs2_pts * ys2_pts)

        denom2 = (N2 * S_yy2 - S_y2 * S_y2)

        if abs(denom2) > 1e-6:

            b2 = (N2 * S_xy2 - S_y2 * S_x2) / denom2
            a2 = (S_x2 - b2 * S_y2) / N2

            angle2_rad = math.atan(b2)
            angle2_deg = math.degrees(angle2_rad)

            y0_2 = 0
            y1_2 = H2 - 1
            x0_2 = int(a2 + b2 * y0_2)
            x1_2 = int(a2 + b2 * y1_2)

            print(f"zy plane: Angle: {angle2_deg:.2f} degrees | x-intercept: {x0_2}")

    # --- Display (rate limited) ---
    if time.time() - last_display > display_period:

        display1 = np.clip(surface1, 0, 5)
        display1 = (display1 / 5.0 * 255).astype(np.uint8)

        display2 = np.clip(surface2, 0, 5)
        display2 = (display2 / 5.0 * 255).astype(np.uint8)

        # draw fitted line if valid
        if 'x0_1' in locals() and 'x1_1' in locals():
            cv.line(display1, (x0_1, y0_1), (x1_1, y1_1), 255, 2)

        if 'x0_2' in locals() and 'x1_2' in locals():
            cv.line(display2, (x0_2, y0_2), (x1_2, y1_2), 255, 2)

        cv.imshow("DVS Angle Estimation xy plane", display1)
        cv.imshow("DVS Angle Estimation zy plane", display2)
        
        cv.waitKey(1)

        last_display = time.time()