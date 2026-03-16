"""
Shared composite layout: one image with banner + panels for experiment, calibration, and DVS preview.

All consumers create one cv2 window, build their frames/workspace, call build_composite(), then a single imshow.
No window creation or imshow in this module; pure numpy/cv2 drawing.
"""

import cv2
import numpy as np

# Used by all callers for consistent dimensions
CAM_WIDTH = 346
CAM_HEIGHT = 260
WORKSPACE_SIZE = 350
GAP = 55
BANNER_HEIGHT = 40
ONE_PANEL_MARGIN = 20

# Default initial window size (pixels) when using WINDOW_NORMAL. Set and forget; not exposed in main.
# Chosen to fit inside 1920×1080 while preserving content aspect ratio.
# Layout: 2-panel = cams only, 3-panel = cams + workspace, 1-panel = workspace only.
DEFAULT_WINDOW_WIDTH_2PANEL = 1920   # content 747×300 → scale to width 1920, height 770
DEFAULT_WINDOW_HEIGHT_2PANEL = 770
DEFAULT_WINDOW_WIDTH_3PANEL = 1920   # content 1152×390 → scale to width 1920, height 650
DEFAULT_WINDOW_HEIGHT_3PANEL = 650
DEFAULT_WINDOW_WIDTH_1PANEL = 1080   # content 390×390 → square limited by 1080 height
DEFAULT_WINDOW_HEIGHT_1PANEL = 1080


def get_default_window_size(has_cams: bool, has_workspace: bool) -> tuple[int, int]:
    """Return (width, height) for initial cv2.resizeWindow. has_cams = frame1 and frame2 used."""
    if has_cams and has_workspace:
        return (DEFAULT_WINDOW_WIDTH_3PANEL, DEFAULT_WINDOW_HEIGHT_3PANEL)
    if has_cams:
        return (DEFAULT_WINDOW_WIDTH_2PANEL, DEFAULT_WINDOW_HEIGHT_2PANEL)
    if has_workspace:
        return (DEFAULT_WINDOW_WIDTH_1PANEL, DEFAULT_WINDOW_HEIGHT_1PANEL)
    return (800, 400)  # fallback


def build_composite(
    title: str,
    frame1: np.ndarray | None = None,
    frame2: np.ndarray | None = None,
    workspace_canvas: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a single BGR image: banner at top, then panels in a row.

    Supported combinations:
      - (frame1, frame2, workspace_canvas) → 3 panels (experiment, calibration)
      - (frame1, frame2) → 2 panels (visualize_dvs_cams)
      - (workspace_canvas only) → 1 panel (e.g. servo_workspace_picker)

    Inputs: title (str); frame1/frame2 BGR (CAM_HEIGHT, CAM_WIDTH, 3) or None;
            workspace_canvas BGR (WORKSPACE_SIZE, WORKSPACE_SIZE, 3) or None.
    Returns: one BGR uint8 image.
    """
    has_cams = frame1 is not None and frame2 is not None
    has_workspace = workspace_canvas is not None

    if has_cams and has_workspace:
        # 3-panel: cam1 | cam2 | workspace
        row_height = max(CAM_HEIGHT, WORKSPACE_SIZE)
        total_width = 2 * CAM_WIDTH + 2 * GAP + WORKSPACE_SIZE
    elif has_cams:
        # 2-panel: cam1 | cam2
        row_height = CAM_HEIGHT
        total_width = 2 * CAM_WIDTH + GAP
    elif has_workspace:
        # 1-panel: workspace only
        row_height = WORKSPACE_SIZE
        total_width = WORKSPACE_SIZE + 2 * ONE_PANEL_MARGIN
    else:
        # Fallback: minimal canvas with just banner
        row_height = 1
        total_width = 400

    total_height = BANNER_HEIGHT + row_height
    composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    composite[:] = (40, 40, 40)

    # Banner
    composite[:BANNER_HEIGHT, :] = (50, 50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
    tx = max(8, (total_width - tw) // 2)
    ty = (BANNER_HEIGHT + th) // 2 + 2
    cv2.putText(composite, title, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    row_y = BANNER_HEIGHT

    if has_cams:
        # Ensure frames are correct size (BGR)
        f1 = frame1 if frame1.shape[:2] == (CAM_HEIGHT, CAM_WIDTH) else cv2.resize(frame1, (CAM_WIDTH, CAM_HEIGHT))
        f2 = frame2 if frame2.shape[:2] == (CAM_HEIGHT, CAM_WIDTH) else cv2.resize(frame2, (CAM_WIDTH, CAM_HEIGHT))
        if len(f1.shape) == 2:
            f1 = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
        if len(f2.shape) == 2:
            f2 = cv2.cvtColor(f2, cv2.COLOR_GRAY2BGR)
        composite[row_y : row_y + CAM_HEIGHT, 0:CAM_WIDTH] = f1
        composite[row_y : row_y + CAM_HEIGHT, CAM_WIDTH + GAP : CAM_WIDTH + GAP + CAM_WIDTH] = f2

    if has_workspace:
        ws = workspace_canvas
        if ws.shape[0] != WORKSPACE_SIZE or ws.shape[1] != WORKSPACE_SIZE:
            ws = cv2.resize(workspace_canvas, (WORKSPACE_SIZE, WORKSPACE_SIZE))
        if len(ws.shape) == 2:
            ws = cv2.cvtColor(ws, cv2.COLOR_GRAY2BGR)
        # Center workspace vertically in row
        wy = row_y + (row_height - WORKSPACE_SIZE) // 2
        if has_cams:
            wx = 2 * CAM_WIDTH + 2 * GAP
        else:
            wx = ONE_PANEL_MARGIN
        composite[wy : wy + WORKSPACE_SIZE, wx : wx + WORKSPACE_SIZE] = ws

    return composite
