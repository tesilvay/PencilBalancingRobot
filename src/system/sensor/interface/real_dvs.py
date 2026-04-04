from dataclasses import dataclass
import threading
import time

import numpy as np
from src.shared import (
    CameraObservation,
    CameraPair,
    CameraParams,
    Measurement
)

from .base import VisionModelBase
from src.system.sensor.observation_model.camera_model import CameraModel
from src.system.sensor.reader.dvs_camera_reader import (
    DVSReader,
)
from src.system.sensor.algo.dvs_algorithms import mask_events_below_line


@dataclass
class RealDVSParams:
    cam_params:               CameraParams
    algo:                     object
    obs_model:                object
    cam1_device:              str | None = None
    cam2_device:              str | None = None
    noise_filter_duration_ms: float | None = None


REAL_DVS_PRESETS = {
    "hough": {
        "cam_params":               "default:default",
        "algo":                     "hough:default",
        "obs_model":                "simple:default",
        "noise_filter_duration_ms": None,
        "cam1_device":              None,
        "cam2_device":              None,
    },
    "sam": {
        "base":                     "hough",
        "algo":                     "sam:default",
        "noise_filter_duration_ms": 5,
    },
}


class RealEventCameraInterface(VisionModelBase):

    def __init__(self, params: RealDVSParams):
        import copy
        p = params
        
        #cam = params.cam_params
        
        self.cam_height_px = p.cam_params.DAVIS346_HEIGHT
        self.cam_width_px = p.cam_params.DAVIS346_WIDTH

        self.cam1_algo = copy.deepcopy(p.algo)
        self.cam2_algo = copy.deepcopy(p.algo)

        self.cam = CameraModel()

        self.dvs_regression_model = p.obs_model
  
        noise_filter_duration_ms = p.noise_filter_duration_ms
        
        cam1_device = p.cam1_device
        cam2_device = p.cam2_device
        
        self._dvs_mask_line_y_cam1 = int(p.cam_params.y_mask_line_1)
        self._dvs_mask_line_y_cam2 = int(p.cam_params.y_mask_line_2)

        self._reader1 = DVSReader(cam1_device, noise_filter_duration_ms=noise_filter_duration_ms)
        self._reader2 = DVSReader(cam2_device, noise_filter_duration_ms=noise_filter_duration_ms)

        self._latest1: CameraObservation | None = None
        self._latest2: CameraObservation | None = None
        
        self._surface1 = np.zeros((self.cam_height_px, self.cam_width_px), dtype=np.float32)
        self._surface2 = np.zeros((self.cam_height_px, self.cam_width_px), dtype=np.float32)
        
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
        mask_y = self._dvs_mask_line_y_cam1 if _cam_id == 1 else self._dvs_mask_line_y_cam2
        while not self._stop.is_set() and reader.is_running():
            batches = []
            while True:
                b = reader.get_event_batch()
                if b is None or len(b) == 0:
                    break
                batches.append(b)

            if batches:
                events = np.concatenate(batches)
                events = mask_events_below_line(events, mask_line_y=mask_y, frame_height=self.cam_height_px)
                surface *= self._decay_display
                if len(events) > 0:
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

    def get_event_accumulator_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Alias for :meth:`get_surfaces` — decaying event-accumulator images for display."""
        return self.get_surfaces()

    def get_y(self, state_true=None) -> Measurement:
        cams_raw = self.get_z()
        
        y_meas = self.cams_to_measurement(cams_px=cams_raw)
        
        return y_meas
        
    def get_z(self) -> CameraPair | None:
        """
        Return latest CameraPair(Pixels) from Hough
        """
        with self._lock:
            obs1_px = self._latest1
            obs2_px = self._latest2

        if obs1_px is None or obs2_px is None:
            print("No observation vision.py line 226")
            return None

        return CameraPair(
            CameraObservation(slope=obs1_px.slope, intercept=obs1_px.intercept),
            CameraObservation(slope=obs2_px.slope, intercept=obs2_px.intercept),
        )

    def cams_to_measurement(self, cams_px):
            
        y_meas = self.dvs_regression_model.estimate(cams_px)

        if super.is_valid_pose(y_meas):
            return y_meas

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
