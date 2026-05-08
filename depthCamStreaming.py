import os
import sys

# Change this path to the folder where you put the realsense2.dll
os.add_dll_directory(os.getcwd()) 

import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import json
import time
import datetime
import matplotlib.pyplot as plt
from collections import deque

# --- SETTINGS ---
TAG_SIZE = 0.150          # Tag 0 (reference base) physical size: 15 cm
TAG_SIZE_TRACKING = 0.03 # Tags 1 & 2 (device) physical size: 2.5 cm
CSV_NAME = "heart_sim_output.csv"
TARGET_FPS = 10            # Consistent output frame rate written to CSV (frames/sec)
ENABLE_PLOT = False       # Temporary: verify relative 3D pose of markers and midpoint
PLOT_UPDATE_HZ = TARGET_FPS  # Plot redraw target; can be overridden by sync-to-recording mode
SYNC_DRAW_TO_RECORDING = True  # Keep display/plot updates aligned to CSV write cadence
ENABLE_PERF_LOG = False   # Print per-stage timing so bottlenecks are visible in terminal
PERF_LOG_INTERVAL_S = 2.0 # Seconds between profiler summaries
ARUCO_DETECT_SCALE = 0.5  # Run tag detection on downscaled frame, then scale corners back up
FAST_ARUCO_MODE = True    # Relax expensive ArUco options to recover real-time performance
WARMUP_S = 2.0            # Seconds of pre-detection before CSV recording starts (pre-warms colour cache)
USE_TAG0_YZ_TO_XY_REMAP = False  # True when Tag 0 is physically mounted on the YZ plane
MARKER_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marker_hsv_config.json")
# Rotate Tag 0 frame back to XY-plane convention when remap is enabled.
# NOTE: This is the transpose (inverse) of the forward rotation to undo YZ mounting.
R_YZ_TO_XY = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=float)

# Correction for camera frame orientation: 90° CCW about Y, 90° CCW about X, then 180° CCW about X
# Composed matrix result
R_CAMERA_FLIP = np.array([
    [1,  0,  0],  # X_plot <- X_actual
    [0,  0,  1],  # Y_plot <- Z_actual
    [0, -1,  0],  # Z_plot <- -Y_actual
], dtype=float)
# --- MARKER COLOUR CONFIG ---
# Set 'enabled': True/False here to choose which colours are tracked.
# HSV thresholds can also be tuned interactively at runtime with the 't' key.
MARKERS = {
    'Purple': { 'enabled': False,
                'hsv_low':  np.array([120, 120,  80]),   # H 128-152, S≥130, V 80-200
                'hsv_high': np.array([165, 155, 175]),
                'bgr': (128, 0, 128) },
    'Pink'  : { 'enabled': True,
                'hsv_low':  np.array([131, 104, 165]),   # H 168-180, S≥100, V≥140
                'hsv_high': np.array([169, 134, 197]),
                'bgr': (147, 20, 255) },
    'Green' : { 'enabled': True,
                'hsv_low':  np.array([ 38, 77,  128]),   # H 58-85, S≥130, V≥30  (dark green)
                'hsv_high': np.array([ 68, 166, 146]),
                'bgr': (0, 255, 0) },
    'Yellow': { 'enabled': False,
                'hsv_low':  np.array([ 24, 150, 100]),   # H 24-32, S≥150, V≥100
                'hsv_high': np.array([ 32, 255, 255]),
                'bgr': (0, 255, 255) },
}


def load_marker_config():
    """Load persisted HSV overrides from disk if available."""
    if not os.path.exists(MARKER_CONFIG_FILE):
        return

    try:
        with open(MARKER_CONFIG_FILE, 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    for name, cfg in saved.items():
        if name not in MARKERS or not isinstance(cfg, dict):
            continue
        low = cfg.get('hsv_low')
        high = cfg.get('hsv_high')
        if isinstance(low, list) and len(low) == 3:
            MARKERS[name]['hsv_low'] = np.array(low, dtype=int)
        if isinstance(high, list) and len(high) == 3:
            MARKERS[name]['hsv_high'] = np.array(high, dtype=int)
        if 'enabled' in cfg:
            MARKERS[name]['enabled'] = bool(cfg['enabled'])


def save_marker_config():
    """Persist current HSV thresholds so tuner saves survive restarts."""
    saved = {}
    for name, cfg in MARKERS.items():
        saved[name] = {
            'enabled': bool(cfg.get('enabled', True)),
            'hsv_low': [int(v) for v in cfg['hsv_low']],
            'hsv_high': [int(v) for v in cfg['hsv_high']],
        }

    with open(MARKER_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(saved, f, indent=2)


def build_timestamped_csv_path(base_name):
    """Append a Unix timestamp to the CSV filename so recordings are not overwritten."""
    root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".csv"
    stamp = str(int(time.time()))
    return f"{root}_{stamp}{ext}"


load_marker_config()

# Global variable for clicked point
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def build_color_mask(hsv_img, hsv_low, hsv_high):
    """Build the same cleaned HSV mask used by runtime marker detection."""
    hsv_smooth = cv2.GaussianBlur(hsv_img, (5, 5), 0)
    mask = cv2.inRange(hsv_smooth, hsv_low, hsv_high)

    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel_close, iterations=2)
    mask = cv2.erode(mask, kernel_close, iterations=2)
    return mask


def hsv_tuner(pipeline, align, intr, depth_scale, initial_low=None, initial_high=None, color_name="Marker"):
    """
    Interactive HSV threshold tuner with live camera feed and trackbars.

    Controls:
      Trackbars — adjust H/S/V min and max in real time
      s         — save and return current thresholds
      r         — reset trackbars to initial values
      q / ESC   — quit without saving (returns None)

    Returns:
        (hsv_low, hsv_high) numpy arrays, or None if cancelled.
    """
    WIN_ORIG  = f"HSV Tuner: {color_name} — Original"
    WIN_MASK  = f"HSV Tuner: {color_name} — Mask"
    WIN_RESULT = f"HSV Tuner: {color_name} — Result (filtered)"
    WIN_CTRL  = f"HSV Tuner: {color_name} — Controls (s=save  r=reset  q=quit)"

    # Default starting values if none supplied
    low  = initial_low.copy()  if initial_low  is not None else np.array([0,   0,   0])
    high = initial_high.copy() if initial_high is not None else np.array([179, 255, 255])

    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 600, 300)

    def nothing(_): pass

    cv2.createTrackbar("H min", WIN_CTRL, int(low[0]),  179, nothing)
    cv2.createTrackbar("H max", WIN_CTRL, int(high[0]), 179, nothing)
    cv2.createTrackbar("S min", WIN_CTRL, int(low[1]),  255, nothing)
    cv2.createTrackbar("S max", WIN_CTRL, int(high[1]), 255, nothing)
    cv2.createTrackbar("V min", WIN_CTRL, int(low[2]),  255, nothing)
    cv2.createTrackbar("V max", WIN_CTRL, int(high[2]), 255, nothing)

    result = None

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

        # Read current trackbar positions
        h_min = cv2.getTrackbarPos("H min", WIN_CTRL)
        h_max = cv2.getTrackbarPos("H max", WIN_CTRL)
        s_min = cv2.getTrackbarPos("S min", WIN_CTRL)
        s_max = cv2.getTrackbarPos("S max", WIN_CTRL)
        v_min = cv2.getTrackbarPos("V min", WIN_CTRL)
        v_max = cv2.getTrackbarPos("V max", WIN_CTRL)

        cur_low  = np.array([h_min, s_min, v_min])
        cur_high = np.array([h_max, s_max, v_max])

        mask_clean = build_color_mask(hsv, cur_low, cur_high)

        filtered = cv2.bitwise_and(img, img, mask=mask_clean)
        runtime_hits = []
        if depth_image is not None:
            runtime_hits = detect_color_markers(
                img.copy(), hsv, depth_image, depth_scale, intr,
                cur_low, cur_high, color_name, (0, 255, 0)
            )
            for _xyz, (px, py) in runtime_hits:
                cv2.circle(filtered, (px, py), 10, (255, 255, 255), 2)

        # Overlay current HSV values on the control window
        info = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(info, f"Color: {color_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info, f"Low : H={h_min:3d}  S={s_min:3d}  V={v_min:3d}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info, f"High: H={h_max:3d}  S={s_max:3d}  V={v_max:3d}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Count detected blobs
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        blob_count = sum(1 for lbl in range(1, num_labels) if stats[lbl, cv2.CC_STAT_AREA] >= 20)
        cv2.putText(info, f"Blobs detected: {blob_count}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info, f"Runtime accepted: {len(runtime_hits)}", (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 220, 0) if runtime_hits else (0, 120, 255), 2)
        cv2.putText(info, "s=SAVE   r=RESET   q=QUIT", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

        cv2.imshow(WIN_ORIG,   img)
        cv2.imshow(WIN_MASK,   mask_clean)
        cv2.imshow(WIN_RESULT, filtered)
        cv2.imshow(WIN_CTRL,   info)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            result = (cur_low.copy(), cur_high.copy())
            break
        elif key == ord('r'):
            cv2.setTrackbarPos("H min", WIN_CTRL, int(low[0]))
            cv2.setTrackbarPos("H max", WIN_CTRL, int(high[0]))
            cv2.setTrackbarPos("S min", WIN_CTRL, int(low[1]))
            cv2.setTrackbarPos("S max", WIN_CTRL, int(high[1]))
            cv2.setTrackbarPos("V min", WIN_CTRL, int(low[2]))
            cv2.setTrackbarPos("V max", WIN_CTRL, int(high[2]))
        elif key in (ord('q'), 27):  # q or ESC
            break

    cv2.destroyWindow(WIN_ORIG)
    cv2.destroyWindow(WIN_MASK)
    cv2.destroyWindow(WIN_RESULT)
    cv2.destroyWindow(WIN_CTRL)
    return result


def detect_color_markers(img, hsv_img, depth_image, depth_scale, intr, hsv_low, hsv_high, color_name, marker_color):
    """
    Detect markers of a specific color using connected components.

    Args:
        img: BGR image
        hsv_img: HSV image
        depth_image: Raw uint16 depth frame as numpy array (from depth_frame.get_data())
        depth_scale: Depth scale factor (metres per unit)
        intr: Camera intrinsics
        hsv_low: Lower HSV bound (numpy array)
        hsv_high: Upper HSV bound (numpy array)
        color_name: Name of color for labeling
        marker_color: BGR tuple for visualization circle color

    Returns:
        List of 3D points [x, y, z] in camera frame (metres)
    """
    mask = build_color_mask(hsv_img, hsv_low, hsv_high)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    marker_points = []
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        if area < 20:
            continue

        # Sample depth with an expanding patch — handles edge regions where
        # aligned depth has sparse/invalid pixels due to sensor offset.
        h_d, w_d = depth_image.shape[:2]
        dist = None
        for PATCH_RADIUS in (10, 20, 35):
            y1 = max(0, cy - PATCH_RADIUS)
            y2 = min(h_d, cy + PATCH_RADIUS + 1)
            x1 = max(0, cx - PATCH_RADIUS)
            x2 = min(w_d, cx + PATCH_RADIUS + 1)
            patch_m = depth_image[y1:y2, x1:x2] * depth_scale
            valid = patch_m[(patch_m > 0.05) & (patch_m <= 3.0)]
            if len(valid) >= 3:          # need at least 3 valid pixels for a reliable median
                dist = float(np.median(valid))
                break
        if dist is None:
            continue

        if dist <= 3.0:
            xyz = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
            marker_points.append((xyz, (cx, cy), area))  # store 3D point + pixel centroid + area

    # Return only the largest blob per call (most likely the real sticker, not a false positive)
    if marker_points:
        marker_points.sort(key=lambda p: p[2], reverse=True)  # sort by area descending
        return [(p[0], p[1]) for p in marker_points[:1]]  # return only best candidate
    return marker_points

def get_multi_sticker_centroid(sticker_points):
    """
    Take the XYZ coordinates of the 2 markers and calculate the midpoint in 3D space.
    """
    if len(sticker_points) >= 2:
        # Sort by Y-coordinate (assuming smaller Y is 'higher')
        sticker_points.sort(key=lambda p: p[1])
        base_pair = sticker_points[:2]
        midpoint = np.mean(base_pair, axis=0)
        return midpoint
    return None

# --- 3. TRANSFORMATION MATRIX COMPUTATION --- (this function is not called))
def compute_world_matrix(corners, ids, intr):
    if ids is None or 0 not in ids: return None
    idx = np.where(ids == 0)[0][0]
    mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[idx], TAG_SIZE, mtx, np.zeros(5))
    
    rmat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = tvec.flatten()
    return T

# --- 4. APPLY TRANSFORMATION & 5. WRITE TO FILE --- this function is not called either
def transform_and_log(p_camera, T_base):
    """
    Transform point from camera frame to AprilTag frame.
    Log to CSV and return transformed coordinates in mm.
    """
    T_inv = np.linalg.inv(T_base)
    p_homog = np.append(p_camera, 1.0)
    p_base = T_inv @ p_homog
    
    # Final coordinates relative to AprilTag in mm
    coords_mm = p_base[:3] * 1000
    return coords_mm

def draw_axes_at_point(img, p_camera, cam_mtx, dist_coeffs, rvec_ref, axis_length=0.03):
    """
    Draw coordinate axes at a 3D point in camera frame.
    Uses the same orientation as the reference frame (AprilTag).
    """
    # tvec is the position of the point in camera frame
    tvec = np.array(p_camera).reshape(3, 1)
    # Use rvec from the reference frame (AprilTag)
    cv2.drawFrameAxes(img, cam_mtx, dist_coeffs, rvec_ref, tvec, axis_length)
    
    # Draw axis end-point labels (Xm, Ym, Zm for "marker" or midpoint)
    axis_pts = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]], dtype=np.float32)
    img_pts, _ = cv2.projectPoints(axis_pts, rvec_ref, tvec, cam_mtx, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    cv2.putText(img, "Xm", tuple(img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, "Ym", tuple(img_pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, "Zm", tuple(img_pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def detect_base_tag(img, intr, corners, ids):
    """
    Detect the base AprilTag (id=0) from pre-computed corners/ids,
    draw markers, axes, and labels on img, and return the transformation matrix.
    """
    if ids is None or 0 not in ids:
        return None

    idx = np.where(ids == 0)[0][0]
    # Draw only the base tag (tag 0)
    cv2.aruco.drawDetectedMarkers(img, [corners[idx]], np.array([[0]]))

    cam_mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)

    # Pose estimation
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[idx], TAG_SIZE, cam_mtx, dist_coeffs)

    # Draw axes at the Tag's location
    cv2.drawFrameAxes(img, cam_mtx, dist_coeffs, rvecs[0], tvecs[0], 0.03)

    # Draw Labels for the World Frame (Tag)
    tag_axis_pts = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], dtype=np.float32)
    tag_img_pts, _ = cv2.projectPoints(tag_axis_pts, rvecs[0], tvecs[0], cam_mtx, dist_coeffs)
    tag_img_pts = tag_img_pts.reshape(-1, 2).astype(int)
    cv2.putText(img, "Xw", tuple(tag_img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Yw", tuple(tag_img_pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Zw", tuple(tag_img_pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Compute transformation matrix
    # Flatten to (3,) — estimatePoseSingleMarkers returns rvecs/tvecs shaped (1,1,3);
    # using rvecs[0] gives (1,3) which can silently corrupt Rodrigues output.
    rvec_flat = rvecs[0].flatten()   # (3,)
    tvec_flat = tvecs[0].flatten()   # (3,)
    rmat, _ = cv2.Rodrigues(rvec_flat)
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = tvec_flat

    return T, rvec_flat


def detect_other_tags(img, intr, base_T, base_rvec, corners, ids):
    """
    Detect AprilTags 1 and 2 from pre-computed corners/ids, compute their midpoint,
    and place the base coordinate system (from tag 0) at this midpoint.
    """
    TARGET_TAG_IDS = [1, 2]

    if ids is None:
        return None, None
     
    # Filter to only process tags we care about (1 and 2)
    filtered_corners = []
    filtered_ids = []
    
    for i, tag_id in enumerate(ids.flatten()):
        if tag_id in TARGET_TAG_IDS:
            filtered_corners.append(corners[i])
            filtered_ids.append(tag_id)
    
    if not filtered_ids:
        return None, None

    cam_mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)

    # Acceptable depth range for Tags 1 & 2 (metres)
    MIN_TAG_DIST = 0.05
    MAX_TAG_DIST = 3.0

    tag_positions = {}

    for i, tag_id in enumerate(filtered_ids):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(filtered_corners[i], TAG_SIZE_TRACKING, cam_mtx, dist_coeffs)
        dist_from_camera = float(np.linalg.norm(tvecs[0]))

        # Reject implausible poses
        if not (MIN_TAG_DIST <= dist_from_camera <= MAX_TAG_DIST):
            continue

        tag_positions[tag_id] = tvecs[0].flatten().copy()  # flatten to (3,) + copy to avoid numpy view aliasing

        # Draw bounding box and label for target tags only
        cv2.aruco.drawDetectedMarkers(img, [filtered_corners[i]], np.array([[tag_id]]))

    if 1 in tag_positions and 2 in tag_positions:
        midpoint = compute_midpoint(tag_positions[1], tag_positions[2])

        # 1. Get the rotation of Tag 0
        rmat_world, _ = cv2.Rodrigues(base_rvec)

        # CORRECTION: Tag 0 is now mounted on the YZ-plane (vertical, orthogonal to original).
        # Pre-multiply rmat_world by the inverse of the extra physical rotation so that the
        # tag frame is brought back to the original XY-plane convention BEFORE any calculation.
        # Tag was tipped +90° around its Y axis (from flat to standing) →  undo with -90° around Y.
        # NOTE: if any output axis appears negated in testing, flip the sign of that column here.
        if USE_TAG0_YZ_TO_XY_REMAP:
            rmat_world = rmat_world @ R_YZ_TO_XY   # corrected tag frame, expressed in camera coords

        # 2. Invert it (Transpose) to get the 'un-rotate' tool
        rmat_inv = rmat_world.T

        # 3. Get the raw distance (Translation)
        tag0_position = base_T[:3, 3]
        p_midpoint = midpoint.flatten()
        raw_dist = p_midpoint - tag0_position

        # 4. Apply the rotation to the distance vector
        # Everything below this point is identical to the original XY-plane logic
        aligned_dist = rmat_inv @ raw_dist

        # Log the properly aligned coordinates
        coords_mm = aligned_dist * 1000

        # Draw coordinate system using cv2.drawFrameAxes - this preserves natural perspective
        # The axes ARE parallel in 3D world space, perspective distortion makes them look non-parallel
        cam_mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        dist_coeffs = np.zeros(5)
        
        # Use Tag 0's exact rotation vector at midpoint position (pure translation of coordinate system)
        midpoint_tvec = midpoint.reshape(1, 3)
        cv2.drawFrameAxes(img, cam_mtx, dist_coeffs, base_rvec.reshape(3, 1), midpoint_tvec, 0.12)

        return coords_mm, tag_positions

    return None, None

def compute_midpoint(tag1_tvec, tag2_tvec):
    """
    Compute the midpoint between two translation vectors.
    """
    return (tag1_tvec + tag2_tvec) / 2


def detect_markers_scaled(detector, img_bgr, scale=1.0):
    """
    Detect ArUco/AprilTags on a downscaled grayscale image for speed,
    then map detected corners back to original pixel coordinates.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if scale is None or scale >= 0.999:
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    h, w = gray.shape[:2]
    small_w = max(64, int(w * scale))
    small_h = max(36, int(h * scale))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    corners, ids, rejected = detector.detectMarkers(small)

    if corners:
        inv_scale = 1.0 / scale
        corners = [c.astype(np.float32) * inv_scale for c in corners]

    return corners, ids, rejected


# --- THE MAIN FUNCTION (THE CONDUCTOR) ---
def main():
    # Setup
    csv_output_path = build_timestamped_csv_path(CSV_NAME)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720,  rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720,  rs.format.z16,  30)
    
    try:
        profile = pipeline.start(config)
        color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_vsp.get_intrinsics()
        capture_fps = color_vsp.fps()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)  # align depth pixels to colour pixel coords
    except Exception as e:
        print(f"Could not start camera: {e}|")
        return

    aruco_params = cv2.aruco.DetectorParameters()
    # Adaptive threshold: narrower search window in fast mode to reduce CPU cost
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23 if FAST_ARUCO_MODE else 53
    aruco_params.adaptiveThreshWinSizeStep = 10 if FAST_ARUCO_MODE else 4
    aruco_params.adaptiveThreshConstant = 3      # lowered from 7 — more sensitive when tag border
                                                 # blends into dark background (no white quiet zone)
    # Allow small markers
    aruco_params.minMarkerPerimeterRate = 0.01
    aruco_params.maxMarkerPerimeterRate = 4.0
    # Lenient polygon approx — handles tags mounted at an angle (trapezoid distortion)
    aruco_params.polygonalApproxAccuracyRate = 0.08
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minMarkerDistanceRate = 0.05
    # Better bit reading for perspective-distorted tags
    aruco_params.perspectiveRemovePixelPerCell = 8
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    # Error correction — 36h11 has strong Hamming distance, use max correction
    aruco_params.errorCorrectionRate = 1.0
    # Subpixel corner refinement for accurate pose
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE if FAST_ARUCO_MODE else cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize = 3 if FAST_ARUCO_MODE else 5
    aruco_params.cornerRefinementMaxIterations = 10 if FAST_ARUCO_MODE else 30
    aruco_params.cornerRefinementMinAccuracy = 0.2 if FAST_ARUCO_MODE else 0.1
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        aruco_params
    )

    # Initialize CSV with mocap-compatible 6-row header
    enabled_markers = [name for name, cfg in MARKERS.items() if cfg.get('enabled', True)]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open(csv_output_path, 'w', newline='') as f:
        w = csv.writer(f)
        # Row 1: file metadata
        w.writerow([
            "Format Version", "1.23",
            "Take Name", os.path.splitext(os.path.basename(csv_output_path))[0],
            "Take Notes", "",
            "Capture Frame Rate", TARGET_FPS,
            "Export Frame Rate", TARGET_FPS,
            "Capture Start Time", now_str,
            "Total Frames in Take", "",
            "Total Exported Frames", "",
            "Rotation Type", "XYZ",
            "Length Units", "Meters",
            "Coordinate Space", "Global",
        ])
        # Row 2: blank
        w.writerow([])
        # Row 3: Type  — Tag0, Tag1, Tag2, Midpoint first, then colour markers
        rigid_cols = ["Tag0", "Tag1", "Tag2", "Midpoint"]
        w.writerow(["", "Type"] + ["Rigid Body Marker", "Rigid Body Marker", "Rigid Body Marker"] * len(rigid_cols)
                                 + ["Marker", "Marker", "Marker"] * len(enabled_markers))
        # Row 4: Name
        name_row = ["", "Name"]
        for n in rigid_cols:
            name_row += [n, n, n]
        for n in enabled_markers:
            name_row += [n, n, n]
        w.writerow(name_row)
        # Row 5: ID
        id_row = ["", "ID"]
        for n in rigid_cols:
            id_row += [n, n, n]
        for n in enabled_markers:
            id_row += [n, n, n]
        w.writerow(id_row)
        # Row 6: sub-type
        sub_row = ["", ""]
        for _ in rigid_cols + enabled_markers:
            sub_row += ["Position", "Position", "Position"]
        w.writerow(sub_row)
        # Row 7: column labels
        col_row = ["Frame", "Time (Seconds)"]
        for _ in rigid_cols + enabled_markers:
            col_row += ["X", "Y", "Z"]
        w.writerow(col_row)

    # Data buffer for plotting
    # coord_buffer = deque(maxlen=300)  # No longer needed - tracking individual markers now

    # Setup matplotlib for live plotting (runs on main thread; updated at low rate)
    if ENABLE_PLOT:
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Device-Relative Tracking (3D Marker Trajectory)")
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.set_xlabel("X (mm)")
        ax_3d.set_ylabel("Y (mm)")
        ax_3d.set_zlabel("Z (mm)")
        ax_3d.set_title("Marker Trajectories")

    try:
        frame_count = 0
        frame_times = deque(maxlen=30)
        write_times = deque(maxlen=30)
        start_time = time.time()
        next_write_due_s = None
        write_interval_s = 1.0 / TARGET_FPS
        last_plot_time = 0.0          # wall-clock time of last 3D plot update
        last_display_circles = {}     # name -> list of (px, py) — drawn every frame for stable display
        # Persistence cache for colour markers: name -> (points_list, timestamp)
        COLOR_PERSISTENCE_S = 1.0   # seconds to hold last known colour position after dropout
        last_good_color = {}        # name -> (points, wall_time)
        TAG_PERSISTENCE_S = 0.5     # seconds to hold last known tag world position after dropout
        last_good_tags = {}         # tag_id -> (world_pos_m, wall_time)
        marker_data = []  # Storage for detected markers
        warmup_end_wall_s = None    # set on first frame; CSV blocked until this time passes

        perf_stats = {}
        perf_last_report = time.perf_counter()

        def perf_add(stage_name, dt_s):
            if not ENABLE_PERF_LOG:
                return
            total_s, count = perf_stats.get(stage_name, (0.0, 0))
            perf_stats[stage_name] = (total_s + dt_s, count + 1)

        def perf_report_if_due():
            nonlocal perf_last_report
            if not ENABLE_PERF_LOG:
                return
            now_perf = time.perf_counter()
            if (now_perf - perf_last_report) < PERF_LOG_INTERVAL_S:
                return
            perf_last_report = now_perf
            if not perf_stats:
                return
            items = []
            for name, (total_s, count) in perf_stats.items():
                avg_ms = (total_s / max(count, 1)) * 1000.0
                items.append((avg_ms, count, name))
            items.sort(reverse=True)
            summary = " | ".join([f"{name}: {avg_ms:.1f}ms ({count}x)" for avg_ms, count, name in items])
            print(f"[Perf] {summary}")
            perf_stats.clear()

        while True:
            loop_t0 = time.perf_counter()

            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            perf_add("wait_for_frames", time.perf_counter() - t0)
            if not color_frame or not depth_frame: continue

            # Warmup: run detection for WARMUP_S seconds before starting CSV recording.
            # This pre-warms last_good_color so frame 0 has full marker data.
            # (The old depth-patch check fired on frame 1, making warmup effectively 0 frames.)
            now_wall_s = time.perf_counter()
            if warmup_end_wall_s is None:
                warmup_end_wall_s = now_wall_s + WARMUP_S
            in_warmup = now_wall_s < warmup_end_wall_s

            if next_write_due_s is None and not in_warmup:
                next_write_due_s = now_wall_s
            slots_due = 0
            if not in_warmup and now_wall_s >= next_write_due_s:
                slots_due = int((now_wall_s - next_write_due_s) // write_interval_s) + 1
                next_write_due_s += slots_due * write_interval_s
            do_write = slots_due > 0

            # Align depth to colour on CSV-write frames AND during warmup (to pre-warm colour cache)
            depth_image = None
            if do_write or in_warmup:
                t0 = time.perf_counter()
                aligned_frames = align.process(frames)
                aligned_depth = aligned_frames.get_depth_frame()
                if aligned_depth:
                    depth_image = np.asanyarray(aligned_depth.get_data())
                perf_add("align_depth", time.perf_counter() - t0)

            img = np.asanyarray(color_frame.get_data())
            display_img = img.copy()  # overlays drawn here; img stays raw for HSV/detection

            # Show warmup countdown on display so user knows recording hasn't started yet
            if in_warmup:
                remaining = max(0.0, (warmup_end_wall_s - now_wall_s))
                cv2.putText(display_img, f"Recording starts in {remaining:.1f}s",
                            (display_img.shape[1]//2 - 220, display_img.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 3)
            # depth_image set above (only when do_write=True)
            # HSV is only needed for colour detection, which is gated on do_write
            hsv = None
            
            # Reset variables for this frame
            tag0_position = None
            rmat_inv = None

            key = cv2.waitKey(1) & 0xFF
            # --- Keypress controls ---
            # t = open HSV tuner menu to select which colour to tune
            # q = quit
            if key == ord('t'):
                # Print colour menu to terminal
                color_keys = list(MARKERS.keys())
                print("\n[HSV Tuner] Select colour to tune:")
                for idx, name in enumerate(color_keys):
                    print(f"  {idx+1}. {name}")
                print("  Enter number (or 0 to cancel): ", end='', flush=True)
                try:
                    choice = int(input())
                except ValueError:
                    choice = 0
                if 1 <= choice <= len(color_keys):
                    selected = color_keys[choice - 1]
                    cfg = MARKERS[selected]
                    result = hsv_tuner(pipeline, align, intr, depth_scale,
                                       initial_low=cfg['hsv_low'].copy(),
                                       initial_high=cfg['hsv_high'].copy(),
                                       color_name=selected)
                    if result is not None:
                        cfg['hsv_low'][:], cfg['hsv_high'][:] = result
                        save_marker_config()
                continue  # skip rest of loop, restart fresh frame

            # 1. Run ArUco detection ONCE for all tags this frame
            t0 = time.perf_counter()
            all_corners, all_ids, _ = detect_markers_scaled(detector, img, ARUCO_DETECT_SCALE)
            perf_add("aruco_detect", time.perf_counter() - t0)

            # 1a. Detect base tag (Tag 0)
            t0 = time.perf_counter()
            result = detect_base_tag(display_img, intr, all_corners, all_ids)
            T_base = None
            rvec_base = None
            if result is not None:
                T_base, rvec_base = result

            # 1b. Detect Tags 1 & 2 using the same detection result
            midpoint_coords = None
            tag_positions = None
            if T_base is not None and rvec_base is not None:
                midpoint_coords, tag_positions = detect_other_tags(display_img, intr, T_base, rvec_base, all_corners, all_ids)
            perf_add("tag_pose", time.perf_counter() - t0)

            # 2. Detect all enabled coloured markers.
            # Runs on do_write frames AND during warmup so last_good_color is pre-populated
            # before the first CSV row is written.
            all_detections = {}  # name -> list of 3D points
            if (do_write or in_warmup) and depth_image is not None:
                t0 = time.perf_counter()
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # compute HSV from raw frame
                for name, cfg in MARKERS.items():
                    if not cfg.get('enabled', True):
                        continue
                    fresh = detect_color_markers(
                        display_img, hsv, depth_image, depth_scale, intr,
                        cfg['hsv_low'], cfg['hsv_high'], name, cfg['bgr'])
                    if fresh:
                        # Good detection — update cache with camera-frame points AND current transform
                        # (only store transform if Tag 0 is currently visible)
                        if rmat_inv is not None and tag0_position is not None:
                            last_good_color[name] = (fresh, time.time(), rmat_inv.copy(), tag0_position.copy())
                        else:
                            last_good_color[name] = (fresh, time.time(), None, None)
                        all_detections[name] = fresh
                    else:
                        # No detection — reuse last known if within persistence window
                        cached = last_good_color.get(name)
                        if cached is not None and (time.time() - cached[1]) <= COLOR_PERSISTENCE_S:
                            all_detections[name] = cached[0]
                        else:
                            all_detections[name] = []
                    # Update persistent pixel positions for stable every-frame display
                    if all_detections[name]:
                        last_display_circles[name] = [(px, py) for _xyz, (px, py) in all_detections[name]]
                perf_add("color_detect", time.perf_counter() - t0)

            # 3. Transform all colored markers relative to Tag 0 world frame
            marker_data = []  # Reset markers for this frame
            tag1_world = None
            tag2_world = None
            midpoint_world = None
            
            if T_base is not None:
                # Tag 0 is visible — always transform relative to it
                tag0_position = T_base[:3, 3]          # (3,) camera-frame position of tag 0
                rmat_world, _ = cv2.Rodrigues(rvec_base.flatten())  # ensure (3,) input
                if USE_TAG0_YZ_TO_XY_REMAP:
                    rmat_world = rmat_world @ R_YZ_TO_XY
                # Apply camera frame axis permutation (map ZX plane to XY)
                rmat_world = rmat_world @ R_CAMERA_FLIP
                rmat_inv = rmat_world.T
                
                if tag_positions is not None and 1 in tag_positions and 2 in tag_positions:
                    # Tags 1 & 2 visible — transform to world frame and cache
                    # tag_positions values are already (3,) after flatten().copy() above
                    tag1_world = rmat_inv @ (tag_positions[1] - tag0_position)
                    tag2_world = rmat_inv @ (tag_positions[2] - tag0_position)
                    midpoint_world = (tag1_world + tag2_world) / 2
                    last_good_tags[1] = (tag1_world, time.time())
                    last_good_tags[2] = (tag2_world, time.time())
                else:
                    # Tags not visible this frame — use cached positions if within persistence window
                    now_t = time.time()
                    c1 = last_good_tags.get(1)
                    c2 = last_good_tags.get(2)
                    if c1 and (now_t - c1[1]) <= TAG_PERSISTENCE_S:
                        tag1_world = c1[0]
                    if c2 and (now_t - c2[1]) <= TAG_PERSISTENCE_S:
                        tag2_world = c2[0]
                    if tag1_world is not None and tag2_world is not None:
                        midpoint_world = (tag1_world + tag2_world) / 2
                
                # All colour markers transformed to Tag 0 world frame
                for name, points in all_detections.items():
                    bgr = MARKERS[name]['bgr']
                    # Use the transform from when this detection was made if available,
                    # otherwise fall back to current frame's transform.
                    cached_entry = last_good_color.get(name)
                    use_rmat_inv = rmat_inv
                    use_tag0 = tag0_position
                    if cached_entry is not None and len(cached_entry) == 4:
                        _, _, cached_rmat_inv, cached_tag0 = cached_entry
                        if cached_rmat_inv is not None and cached_tag0 is not None:
                            use_rmat_inv = cached_rmat_inv
                            use_tag0 = cached_tag0
                    if use_rmat_inv is None or use_tag0 is None:
                        continue  # no valid transform available — skip this marker
                    for marker_point, _px in points:
                        marker_camera = np.array(marker_point)
                        marker_world = use_rmat_inv @ (marker_camera - use_tag0)
                        marker_world_mm = marker_world * 1000
                        marker_data.append((marker_world_mm, bgr, name))
            # If Tag 0 not visible, no transformation possible — markers not plotted

            # Write CSV rows at TARGET_FPS cadence; if slots were missed, write catch-up rows.
            if do_write:
                t0 = time.perf_counter()
                detected_lookup = {}
                for marker_mm, _bgr, _name in marker_data:
                    if _name not in detected_lookup:
                        detected_lookup[_name] = marker_mm
                with open(csv_output_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for _ in range(slots_due):
                        nominal_time = frame_count / TARGET_FPS
                        csv_row = [frame_count, f"{nominal_time:.4f}"]
                        # Tag0 is always world origin (0,0,0) — recorded as reference
                        tag0_present = T_base is not None
                        csv_row += ["0.000000", "0.000000", "0.000000"] if tag0_present else ["", "", ""]
                        for pos_m in [tag1_world, tag2_world, midpoint_world]:
                            if pos_m is not None:
                                csv_row += [f"{pos_m[0]:.6f}", f"{pos_m[1]:.6f}", f"{pos_m[2]:.6f}"]
                            else:
                                csv_row += ["", "", ""]
                        for n in enabled_markers:
                            if n in detected_lookup:
                                pos_m = detected_lookup[n] / 1000.0
                                csv_row += [f"{pos_m[0]:.6f}", f"{pos_m[1]:.6f}", f"{pos_m[2]:.6f}"]
                            else:
                                csv_row += ["", "", ""]
                        writer.writerow(csv_row)
                        frame_count += 1
                    write_times.append(time.time())
                    perf_add("csv_write", time.perf_counter() - t0)

            # Overlay world-frame positions for live verification (matches what goes into CSV)
            dbg_y = 30
            def _fmt(v): return f"{v[0]*1000:.1f}, {v[1]*1000:.1f}, {v[2]*1000:.1f} mm" if v is not None else "---"
            for lbl, val in [("Tag1", tag1_world), ("Tag2", tag2_world), ("Mid", midpoint_world)]:
                cv2.putText(display_img, f"{lbl}: {_fmt(val)}",
                            (display_img.shape[1] - 480, dbg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                dbg_y += 24

            # Draw colour marker circles every frame using last known pixel positions
            for name, pixels in last_display_circles.items():
                bgr = MARKERS[name]['bgr']
                for (px, py) in pixels:
                    if 0 <= px < display_img.shape[1] and 0 <= py < display_img.shape[0]:
                        cv2.circle(display_img, (px, py), 8, bgr, 2)

            # Calculate and display rates (camera loop vs recording cadence)
            frame_times.append(time.time())
            cam_fps = 0.0
            rec_hz = 0.0
            if len(frame_times) > 1:
                cam_fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
            if len(write_times) > 1:
                rec_hz = (len(write_times) - 1) / max(write_times[-1] - write_times[0], 1e-6)

            h, w = display_img.shape[:2]
            cv2.putText(display_img, f"Cam FPS: {cam_fps:.1f}", (20, h - 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display_img, f"REC Hz: {rec_hz:.2f} / {TARGET_FPS:.2f}", (20, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            
            if (not SYNC_DRAW_TO_RECORDING) or do_write or in_warmup:
                t0 = time.perf_counter()
                cv2.imshow("Real-Time Heart Tracker", display_img)
                perf_add("display", time.perf_counter() - t0)

            # Update 3D plot at a limited cadence so Matplotlib does not throttle acquisition.
            plot_tick_due = do_write if SYNC_DRAW_TO_RECORDING else ((time.time() - last_plot_time) >= (1.0 / max(PLOT_UPDATE_HZ, 0.1)))
            if ENABLE_PLOT and plot_tick_due:
                t0 = time.perf_counter()
                last_plot_time = time.time()
                ax_3d.clear()
                ax_3d.scatter(0, 0, 0, color='black', s=120, marker='^', label='Tag 0 (Origin)', alpha=1.0)

                if tag1_world is not None:
                    t1 = tag1_world * 1000
                    ax_3d.scatter(t1[0], t1[1], t1[2], color='orange', s=120, marker='D', label='Tag 1', alpha=1.0)

                if tag2_world is not None:
                    t2 = tag2_world * 1000
                    ax_3d.scatter(t2[0], t2[1], t2[2], color='magenta', s=120, marker='D', label='Tag 2', alpha=1.0)

                if midpoint_world is not None:
                    mid = midpoint_world * 1000
                    ax_3d.scatter(mid[0], mid[1], mid[2], color='red', s=100, marker='o', label='Midpoint', alpha=1.0)
                    if tag1_world is not None and tag2_world is not None:
                        t1 = tag1_world * 1000; t2 = tag2_world * 1000
                        ax_3d.plot([t1[0], t2[0]], [t1[1], t2[1]], [t1[2], t2[2]],
                                   'm--', alpha=0.7, linewidth=2, label='Tag1-Tag2 Line')

                for marker_mm, color_bgr, color_name in marker_data:
                    color_norm = (color_bgr[2]/255.0, color_bgr[1]/255.0, color_bgr[0]/255.0)
                    ax_3d.scatter(marker_mm[0], marker_mm[1], marker_mm[2],
                                  color=color_norm, s=80, marker='s', label=color_name, alpha=0.8)

                axis_length, axis_thickness = 200, 3
                ax_3d.quiver(0, 0, 0, axis_length, 0, 0,          color='red',   alpha=0.8, arrow_length_ratio=0.05, linewidth=axis_thickness, label='X-axis (World)')
                ax_3d.quiver(0, 0, 0, 0, axis_length, 0,          color='green', alpha=0.8, arrow_length_ratio=0.05, linewidth=axis_thickness, label='Y-axis (World)')
                ax_3d.quiver(0, 0, 0, 0, 0,          axis_length, color='blue',  alpha=0.8, arrow_length_ratio=0.05, linewidth=axis_thickness, label='Z-axis (World)')

                ax_3d.set_xlabel("X (mm)"); ax_3d.set_ylabel("Y (mm)"); ax_3d.set_zlabel("Z (mm)")
                ax_3d.set_xlim([-500, 500]); ax_3d.set_ylim([-500, 500]); ax_3d.set_zlim([-500, 500])

                handles, labels_h = ax_3d.get_legend_handles_labels()
                ax_3d.legend(dict(zip(labels_h, handles)).values(), dict(zip(labels_h, handles)).keys(),
                             loc='upper right', fontsize=7)

                tags_detected = []
                if T_base is not None: tags_detected.append('0')
                if tag1_world is not None: tags_detected.append('1')
                if tag2_world is not None: tags_detected.append('2')
                fig.suptitle(f"World Frame Tracking | Markers: {len(marker_data)} "
                             f"| Tags detected: {', '.join(tags_detected) or 'None'}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()  # keep window responsive (non-blocking)
                perf_add("matplotlib", time.perf_counter() - t0)

            perf_add("loop_total", time.perf_counter() - loop_t0)
            perf_report_if_due()

            if key == ord('q'): break

    finally:
        pipeline.stop()
        if ENABLE_PLOT:
            plt.close('all')
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()