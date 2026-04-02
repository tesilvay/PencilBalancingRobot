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
# Single camera row (banner + one CAM_WIDTH panel)
DEFAULT_WINDOW_WIDTH_1CAM = 960
DEFAULT_WINDOW_HEIGHT_1CAM = 380
# One camera + workspace (banner + cam | gap | workspace)
DEFAULT_WINDOW_WIDTH_1CAM_WS = 1280
DEFAULT_WINDOW_HEIGHT_1CAM_WS = 650
# One DVS: cam + right info panel (see :func:`build_one_dvs_composite`)
ONE_DVS_SIDE_PANEL_WIDTH = 400
ONE_DVS_COMPOSITE_WIDTH = ONE_PANEL_MARGIN + CAM_WIDTH + GAP + ONE_DVS_SIDE_PANEL_WIDTH
DEFAULT_WINDOW_WIDTH_1DVS_SIDE = max(960, ONE_DVS_COMPOSITE_WIDTH + 80)
DEFAULT_WINDOW_HEIGHT_1DVS_SIDE = 420


def get_default_window_size(
    has_cams: bool,
    has_workspace: bool,
    *,
    single_cam: bool = False,
    one_dvs_side_panel: bool = False,
) -> tuple[int, int]:
    """Return (width, height) for initial cv2.resizeWindow.

    ``has_cams`` is True when at least one camera panel is shown. Use ``single_cam=True``
    when only one ``(CAM_HEIGHT, CAM_WIDTH)`` panel is used (not the two-cam pair).
    ``one_dvs_side_panel`` matches :func:`build_one_dvs_composite` (wider bitmap).
    """
    if single_cam and has_workspace:
        return (DEFAULT_WINDOW_WIDTH_1CAM_WS, DEFAULT_WINDOW_HEIGHT_1CAM_WS)
    if single_cam and has_cams and one_dvs_side_panel:
        return (DEFAULT_WINDOW_WIDTH_1DVS_SIDE, DEFAULT_WINDOW_HEIGHT_1DVS_SIDE)
    if single_cam and has_cams:
        return (DEFAULT_WINDOW_WIDTH_1CAM, DEFAULT_WINDOW_HEIGHT_1CAM)
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
      - (frame1 only) → 1 camera panel (calibration / OneDvsVisualizer)
      - (frame1, workspace_canvas) → cam + workspace (single-cam workspace modes)
      - (workspace_canvas only) → 1 panel (e.g. servo_workspace_picker)

    Inputs: title (str); frame1/frame2 BGR (CAM_HEIGHT, CAM_WIDTH, 3) or None;
            workspace_canvas BGR (WORKSPACE_SIZE, WORKSPACE_SIZE, 3) or None.
    Returns: one BGR uint8 image.
    """
    has_two_cams = frame1 is not None and frame2 is not None
    has_one_cam = frame1 is not None and frame2 is None
    has_workspace = workspace_canvas is not None

    if has_two_cams and has_workspace:
        # 3-panel: cam1 | cam2 | workspace
        row_height = max(CAM_HEIGHT, WORKSPACE_SIZE)
        total_width = 2 * CAM_WIDTH + 2 * GAP + WORKSPACE_SIZE
    elif has_two_cams:
        # 2-panel: cam1 | cam2
        row_height = CAM_HEIGHT
        total_width = 2 * CAM_WIDTH + GAP
    elif has_one_cam and has_workspace:
        row_height = max(CAM_HEIGHT, WORKSPACE_SIZE)
        total_width = CAM_WIDTH + GAP + WORKSPACE_SIZE
    elif has_one_cam:
        row_height = CAM_HEIGHT
        total_width = CAM_WIDTH + 2 * ONE_PANEL_MARGIN
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

    if has_two_cams:
        # Ensure frames are correct size (BGR)
        f1 = frame1 if frame1.shape[:2] == (CAM_HEIGHT, CAM_WIDTH) else cv2.resize(frame1, (CAM_WIDTH, CAM_HEIGHT))
        f2 = frame2 if frame2.shape[:2] == (CAM_HEIGHT, CAM_WIDTH) else cv2.resize(frame2, (CAM_WIDTH, CAM_HEIGHT))
        if len(f1.shape) == 2:
            f1 = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
        if len(f2.shape) == 2:
            f2 = cv2.cvtColor(f2, cv2.COLOR_GRAY2BGR)
        composite[row_y : row_y + CAM_HEIGHT, 0:CAM_WIDTH] = f1
        composite[row_y : row_y + CAM_HEIGHT, CAM_WIDTH + GAP : CAM_WIDTH + GAP + CAM_WIDTH] = f2
    elif has_one_cam:
        f1 = frame1 if frame1.shape[:2] == (CAM_HEIGHT, CAM_WIDTH) else cv2.resize(frame1, (CAM_WIDTH, CAM_HEIGHT))
        if len(f1.shape) == 2:
            f1 = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
        x0 = ONE_PANEL_MARGIN if not has_workspace else 0
        composite[row_y : row_y + CAM_HEIGHT, x0 : x0 + CAM_WIDTH] = f1

    if has_workspace:
        ws = workspace_canvas
        if ws.shape[0] != WORKSPACE_SIZE or ws.shape[1] != WORKSPACE_SIZE:
            ws = cv2.resize(workspace_canvas, (WORKSPACE_SIZE, WORKSPACE_SIZE))
        if len(ws.shape) == 2:
            ws = cv2.cvtColor(ws, cv2.COLOR_GRAY2BGR)
        # Center workspace vertically in row
        wy = row_y + (row_height - WORKSPACE_SIZE) // 2
        if has_two_cams:
            wx = 2 * CAM_WIDTH + 2 * GAP
        elif has_one_cam:
            wx = CAM_WIDTH + GAP
        else:
            wx = ONE_PANEL_MARGIN
        composite[wy : wy + WORKSPACE_SIZE, wx : wx + WORKSPACE_SIZE] = ws

    return composite


def _wrap_text_lines(
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> list[str]:
    """Word-wrap ``text`` into lines that fit ``max_width`` (pixels)."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            if not lines or lines[-1] != "":
                lines.append("")
            continue
        words = paragraph.split()
        current: list[str] = []
        for w in words:
            test = " ".join(current + [w])
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw <= max_width or not current:
                current.append(w)
            else:
                lines.append(" ".join(current))
                current = [w]
        if current:
            lines.append(" ".join(current))
    return lines if lines else [""]


def build_one_dvs_composite(
    frame1: np.ndarray,
    side_text: str,
    *,
    banner_short: str = "One DVS | Q: quit",
) -> np.ndarray:
    """
    One camera at native ``(CAM_HEIGHT, CAM_WIDTH)`` plus a right-hand panel for long status text.

    The banner shows only ``banner_short``; ``side_text`` is word-wrapped in the side panel.
    Used by :class:`~visualization.realtime_visualizer.OneDvsVisualizer` only.
    """
    total_width = ONE_DVS_COMPOSITE_WIDTH
    row_height = CAM_HEIGHT
    total_height = BANNER_HEIGHT + row_height
    composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    composite[:] = (40, 40, 40)

    composite[:BANNER_HEIGHT, :] = (50, 50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(banner_short, font, font_scale, thickness)
    tx = max(8, (total_width - tw) // 2)
    ty = (BANNER_HEIGHT + th) // 2 + 2
    cv2.putText(composite, banner_short, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    row_y = BANNER_HEIGHT
    f1 = frame1 if frame1.shape[:2] == (CAM_HEIGHT, CAM_WIDTH) else cv2.resize(frame1, (CAM_WIDTH, CAM_HEIGHT))
    if len(f1.shape) == 2:
        f1 = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
    x0 = ONE_PANEL_MARGIN
    composite[row_y : row_y + CAM_HEIGHT, x0 : x0 + CAM_WIDTH] = f1

    x_side = ONE_PANEL_MARGIN + CAM_WIDTH + GAP
    composite[row_y : row_y + CAM_HEIGHT, x_side : x_side + ONE_DVS_SIDE_PANEL_WIDTH] = (42, 42, 42)

    pad = 10
    max_text_width = ONE_DVS_SIDE_PANEL_WIDTH - 2 * pad
    side_font_scale = 0.42
    side_thickness = 1
    (_, line_h), _ = cv2.getTextSize("Hg", font, side_font_scale, side_thickness)
    line_step = int(line_h) + 4
    max_lines = max(1, (CAM_HEIGHT - 2 * pad) // line_step)

    wrapped = _wrap_text_lines(side_text, font, side_font_scale, side_thickness, max_text_width)
    if len(wrapped) > max_lines:
        lines = wrapped[: max_lines - 1]
        remainder = len(wrapped) - len(lines)
        tail = f"... (+{remainder} more line{'s' if remainder != 1 else ''})"
        lines.append(tail)
    else:
        lines = wrapped

    y = row_y + pad + line_h
    for line in lines:
        if y > row_y + CAM_HEIGHT - pad:
            break
        cv2.putText(
            composite,
            line,
            (x_side + pad, y),
            font,
            side_font_scale,
            (220, 220, 220),
            side_thickness,
            lineType=cv2.LINE_AA,
        )
        y += line_step

    return composite
