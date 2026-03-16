import threading
import time
import numpy as np
from collections import deque
from perception.camera_model import CameraModel
from core.sim_types import (
    SystemState,
    CameraParams,
    CameraObservation,
    CameraPair,
    PoseMeasurement
)

def get_measurements(cams: CameraPair):
    b1 = cams.cam1.intercept
    s1 = cams.cam1.slope

    b2 = cams.cam2.intercept
    s2 = cams.cam2.slope
    
    return b1, s1, b2, s2

# ============================================================
# Base Vision Model (shared math)
# ============================================================

class VisionModelBase:

    def __init__(self, camera_params: CameraParams):
        self.xr = camera_params.xr
        self.yr = camera_params.yr

    # -------------------------------------------------
    # Reconstruct 3D pose from two camera observations
    # -------------------------------------------------
    def reconstruct(self, cams: CameraPair) -> PoseMeasurement:

        b1, s1, b2, s2 = get_measurements(cams)

        denom = b1 * b2 + 1.0
        if abs(denom) < 1e-8:
            denom = 1e-8

        X = (b1 * self.yr + b1 * b2 * self.xr) / denom
        Y = (b2 * self.xr - b1 * b2 * self.yr) / denom
        alpha_x = (s1 + b1 * s2) / denom
        alpha_y = (s2 - b2 * s1) / denom

        if getattr(self, "dvs_calibration", None) is not None:
            X_cal, Y_cal = self.dvs_calibration.apply(b1, b2)
            if np.isfinite(X_cal) and np.isfinite(Y_cal):
                X, Y = X_cal, Y_cal
                print(f"Calibrated pose mm: X={X*1000:.2f}, Y={Y*1000:.2f}")

        return PoseMeasurement(
            X=X,
            Y=Y,
            alpha_x=alpha_x,
            alpha_y=alpha_y
        )
        
    def project(self, state_true: SystemState) -> CameraPair:

        X = state_true.x
        Y = state_true.y
        alpha_x = state_true.alpha_x
        alpha_y = state_true.alpha_y

        denom1 = Y + self.yr
        if abs(denom1) < 1e-8:
            denom1 = 1e-8

        b1 = X / denom1
        s1 = alpha_x - (X * alpha_y) / denom1

        denom2 = self.xr - X
        if abs(denom2) < 1e-8:
            denom2 = 1e-8

        b2 = Y / denom2
        s2 = alpha_y + (Y * alpha_x) / denom2

        cam1 = CameraObservation(slope=s1, intercept=b1)
        cam2 = CameraObservation(slope=s2, intercept=b2)

        return CameraPair(cam1=cam1, cam2=cam2)


# ============================================================
# Real DVS Camera Interface
# ============================================================

class RealEventCameraInterface(VisionModelBase):

    def __init__(
        self,
        camera_params,
        cam1_algo,
        cam2_algo,
        cam1_device: str,
        cam2_device: str,
        noise_filter_duration_ms: float | None = None,
    ):
        super().__init__(camera_params)
        self.cam1_algo = cam1_algo
        self.cam2_algo = cam2_algo
        self.cam = CameraModel()

        from perception.dvs_camera_reader import DVSReader, DAVIS346_WIDTH, DAVIS346_HEIGHT

        self._reader1 = DVSReader(cam1_device, noise_filter_duration_ms=noise_filter_duration_ms)
        self._reader2 = DVSReader(cam2_device, noise_filter_duration_ms=noise_filter_duration_ms)

        self._latest1: CameraObservation | None = None
        self._latest2: CameraObservation | None = None
        self._surface1 = np.zeros((DAVIS346_HEIGHT, DAVIS346_WIDTH), dtype=np.float32)
        self._surface2 = np.zeros((DAVIS346_HEIGHT, DAVIS346_WIDTH), dtype=np.float32)
        self._decay_display = 0.5
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread1 = threading.Thread(target=self._reader_loop, args=(self._reader1, self.cam1_algo, 1))
        self._thread2 = threading.Thread(target=self._reader_loop, args=(self._reader2, self.cam2_algo, 2))
        self._thread1.daemon = True
        self._thread2.daemon = True
        self._thread1.start()
        self._thread2.start()

    def _reader_loop(self, reader, algo, _cam_id: int):
        """Background loop: drain all queued batches, update algo, store latest."""
        surface = self._surface1 if _cam_id == 1 else self._surface2
        while not self._stop.is_set() and reader.is_running():
            batches = []
            while True:
                b = reader.get_event_batch()
                if b is None or len(b) == 0:
                    break
                batches.append(b)

            if batches:
                events = np.concatenate(batches)
                surface *= self._decay_display
                np.add.at(surface, (events["y"], events["x"]), 1.0)
                result = algo.update(events)
                if not isinstance(result, tuple):
                    with self._lock:
                        if _cam_id == 1:
                            self._latest1 = result
                        else:
                            self._latest2 = result
            else:
                time.sleep(0.0001)

    def get_surfaces(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copy of current event surfaces for visualization."""
        with self._lock:
            return self._surface1.copy(), self._surface2.copy()

    def get_observation(self, state_true=None) -> CameraPair | None:
        """
        Return latest CameraPair from Hough (same interface as sim).
        state_true is ignored; real cams use background event stream.
        """
        with self._lock:
            obs1_px = self._latest1
            obs2_px = self._latest2

        if obs1_px is None or obs2_px is None:
            return None

        obs1 = self.cam.pixel_to_normalized(obs1_px)
        obs2 = self.cam.pixel_to_normalized(obs2_px)

        return CameraPair(
            CameraObservation(slope=obs1.slope, intercept=obs1.intercept),
            CameraObservation(slope=obs2.slope, intercept=obs2.intercept),
        )

    def reset(self):
        """Reset both Hough algorithms."""
        self.cam1_algo.reset()
        self.cam2_algo.reset()
        with self._lock:
            self._latest1 = None
            self._latest2 = None

    def close(self):
        """Stop reader threads and release cameras."""
        self._stop.set()
        self._thread1.join(timeout=1.0)
        self._thread2.join(timeout=1.0)
        self._reader1.close()
        self._reader2.close()

# ============================================================
# Simulated DVS Camera Interface
# ============================================================
class SimEventCameraInterface(VisionModelBase):

    def __init__(self, camera_params, cam1_algo, cam2_algo):
        super().__init__(camera_params)

        self.cam1_algo = cam1_algo
        self.cam2_algo = cam2_algo
        
        self.cam = CameraModel()

    def generate_events(self, b, s, n=200):

        # convert normalized line → pixel line (x = s*y + b)
        obs_px = self.cam.normalized_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept

        cam_height = self.cam.height
        cam_width = self.cam.width

        # sample points along the line
        ys = np.random.uniform(0, cam_height, n)
        xs = s_px * ys + b_px

        # add pixel noise
        xs += np.random.normal(0, 1.0, n)

        # keep only events inside the image
        mask = (xs >= 0) & (xs < cam_width)
        xs = xs[mask]
        ys = ys[mask]

        xs = xs.astype(np.int16)
        ys = ys.astype(np.int16)

        events = np.zeros(len(xs), dtype=[("x", np.int16), ("y", np.int16)])
        events["x"] = xs
        events["y"] = ys

        return events

    def get_observation(self, state_true):

        # compute true line
        cams = super().project(state_true)

        b1, s1, b2, s2 = get_measurements(cams)

        events1 = self.generate_events(b1, s1)
        events2 = self.generate_events(b2, s2)

        result1 = self.cam1_algo.update(events1)
        result2 = self.cam2_algo.update(events2)

        # tracker not ready yet (returns (None, None) or CameraObservation)
        if isinstance(result1, tuple) or isinstance(result2, tuple):
            return None

        obs1 = self.cam.pixel_to_normalized(result1)
        obs2 = self.cam.pixel_to_normalized(result2)

        return CameraPair(
            CameraObservation(slope=obs1.slope, intercept=obs1.intercept),
            CameraObservation(slope=obs2.slope, intercept=obs2.intercept)
        )
        
    def reset(self):
        self.cam1_algo.reset()
        self.cam2_algo.reset()


# ============================================================
# Simulated Vision Model
# ============================================================

class SimVisionModel(VisionModelBase):

    def __init__(self, camera_params, noise_std=None, delay_steps=0):
        super().__init__(camera_params)

        self.noise_std = noise_std
        self.delay_steps = delay_steps
        self.buffer = deque(maxlen=delay_steps + 1)

    # -------------------------------------------------
    # Project true 3D state into both camera views
    # -------------------------------------------------
    def get_observation(self, state_true: SystemState) -> CameraPair:
        
        cams = super().project(state_true)
        
        noisy_cams = self._add_noise(cams)

        if self.delay_steps > 0:
            self.buffer.append(noisy_cams)

            if len(self.buffer) <= self.delay_steps:
                return noisy_cams

            return self.buffer[0]

        return noisy_cams
    
    def _add_noise(self, cams: CameraPair):
        
        b1, s1, b2, s2 = get_measurements(cams)
        
        if self.noise_std is not None:
            s1 += np.random.normal(0, self.noise_std)
            b1 += np.random.normal(0, self.noise_std)
            s2 += np.random.normal(0, self.noise_std)
            b2 += np.random.normal(0, self.noise_std)
        
        cam1 = CameraObservation(slope=s1, intercept=b1)
        cam2 = CameraObservation(slope=s2, intercept=b2)

        noisy_cams = CameraPair(cam1=cam1, cam2=cam2)
        
        return noisy_cams

    def reset(self):
        self.buffer.clear()