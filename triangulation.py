"""
triangulation.py
----------------
Dual-camera stereo RGB tracking for colour marker 3D position measurement.

Both cameras are Intel RealSense devices used as plain RGB sources — depth
streams are NOT used.  3D positions come from stereo triangulation of matched
2D colour-blob centroids, eliminating depth-sensor noise entirely.

Pipeline (per frame)
--------------------
1. Capture colour frames from both cameras simultaneously.
2. Camera 1 (primary): detect AprilTag 0 → world-frame origin (PnP only, no depth).
3. Camera 1 + Camera 2: detect enabled colour blobs (Pink, Green) independently.
4. For each colour: match the blob seen in cam1 to the blob seen in cam2,
   then triangulate to a 3D point in camera-1 frame.
5. Transform the triangulated point into Tag 0 world frame.
6. Write one CSV row per TARGET_FPS tick — format identical to
   depthCamStreaming.py so dataMatching.py works unchanged.

Calibration (run once)
----------------------
Press 'c' during the live preview to enter stereo calibration mode.
Hold a checkerboard in view of BOTH cameras, move it through different
positions and angles until enough frame pairs are collected, then press
ESC.  Intrinsics + extrinsics are saved to STEREO_CALIB_FILE.
"""

import os
import sys
import json
import time
import datetime
import csv

import queue
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

# ── CAMERA SERIAL NUMBERS ─────────────────────────────────────────────────────
# Run with --list-cameras to print all connected devices and their serials.
# Set these once you know which physical camera is which.
SERIAL_CAM1 = "234422062947"   # primary   — Tag 0 (world reference) detected here
SERIAL_CAM2 = "211622060595"   # secondary — used only for triangulation

# ── PATHS ─────────────────────────────────────────────────────────────────────
DIR              = os.path.dirname(os.path.abspath(__file__))
STEREO_CALIB_FILE = os.path.join(DIR, "stereo_calibration.json")
MARKER_CFG_FILE   = os.path.join(DIR, "marker_hsv_config.json")
CSV_NAME          = "triangulation_output.csv"

# ── APRILTAG SETTINGS ─────────────────────────────────────────────────────────
TAG_SIZE          = 0.150    # Tag 0 physical side length (m)
TAG_SIZE_TRACKING = 0.030    # Tags 1 & 2 physical side length (m)

# ── RECORDING ─────────────────────────────────────────────────────────────────
TARGET_FPS  = 10             # CSV write rate (Hz)
WARMUP_S    = 2.0            # seconds of detection before CSV recording starts
ENABLE_PLOT = True           # Live 3D debug plot (disable for headless / speed)

# ── CAMERA STREAM RESOLUTION ──────────────────────────────────────────────────
# Two cameras simultaneously need ~2× bandwidth.  848×480 is a native D4xx
# resolution and uses ~40 % less bandwidth than 1280×720, which is usually
# needed to keep both cameras alive on the same USB controller.
CAM_WIDTH  = 848
CAM_HEIGHT = 480
CAM_FPS    = 30

# ── STEREO CALIBRATION ────────────────────────────────────────────────────────
CHECKERBOARD    = (9, 6)     # interior corners (columns, rows)
SQUARE_SIZE_M   = 0.025      # physical square size in metres
CALIB_FRAMES    = 30         # number of valid stereo frame pairs to collect

# ── SANITY CHECK ──────────────────────────────────────────────────────────────
# Max 3D error (mm) between PnP and stereo-triangulated tag position before
# an on-screen warning fires.  Consistently high values → redo stereo calib.
STEREO_CHECK_WARN_MM = 30.0

# ── COLOUR MARKERS ────────────────────────────────────────────────────────────
CIRCULARITY_MIN  = 0.45
MARKER_AREA_MIN  = 20
MARKER_AREA_MAX  = 8000
SHADOW_V_SLACK   = 90        # V-channel slack for shadow fallback

MARKERS = {
    "Pink": {
        "enabled":  True,
        "hsv_low":  np.array([131, 104, 165]),
        "hsv_high": np.array([169, 134, 197]),
        "bgr":      (147, 20, 255),
    },
    "Green": {
        "enabled":  True,
        "hsv_low":  np.array([ 38,  77, 128]),
        "hsv_high": np.array([ 68, 166, 146]),
        "bgr":      (0, 255, 0),
    },
}

# ── COORDINATE REMAPPING ──────────────────────────────────────────────────────
# Same axis permutation as depthCamStreaming.py so downstream CSV processing
# is identical.
R_CAMERA_FLIP = np.array([
    [1,  0,  0],   # X_out ← X_cam
    [0,  0,  1],   # Y_out ← Z_cam  (depth direction → becomes "Y" in world)
    [0, -1,  0],   # Z_out ← -Y_cam
], dtype=float)

USE_TAG0_YZ_TO_XY_REMAP = False
R_YZ_TO_XY = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_marker_config():
    if not os.path.exists(MARKER_CFG_FILE):
        return
    try:
        with open(MARKER_CFG_FILE, "r") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    for name, cfg in saved.items():
        if name not in MARKERS:
            continue
        if isinstance(cfg.get("hsv_low"),  list):
            MARKERS[name]["hsv_low"]  = np.array(cfg["hsv_low"],  dtype=int)
        if isinstance(cfg.get("hsv_high"), list):
            MARKERS[name]["hsv_high"] = np.array(cfg["hsv_high"], dtype=int)
        if "enabled" in cfg:
            MARKERS[name]["enabled"] = bool(cfg["enabled"])


def save_marker_config():
    saved = {
        name: {
            "enabled":  bool(cfg.get("enabled", True)),
            "hsv_low":  [int(v) for v in cfg["hsv_low"]],
            "hsv_high": [int(v) for v in cfg["hsv_high"]],
        }
        for name, cfg in MARKERS.items()
    }
    with open(MARKER_CFG_FILE, "w") as f:
        json.dump(saved, f, indent=2)


def load_stereo_calibration():
    """
    Load stereo calibration from JSON.

    Returns
    -------
    (K1, D1, K2, D2, R, T) or None if file not found / invalid.
      K1, K2 : (3,3) intrinsic matrices
      D1, D2 : (5,) distortion coefficients
      R      : (3,3) rotation of cam2 relative to cam1
      T      : (3,)  translation of cam2 relative to cam1 (metres)
    """
    if not os.path.exists(STEREO_CALIB_FILE):
        return None
    try:
        with open(STEREO_CALIB_FILE) as f:
            d = json.load(f)
        K1 = np.array(d["K1"]); D1 = np.array(d["D1"])
        K2 = np.array(d["K2"]); D2 = np.array(d["D2"])
        R  = np.array(d["R"]);  T  = np.array(d["T"]).flatten()
        print(f"[Stereo] Loaded calibration from {STEREO_CALIB_FILE}")
        print(f"  Baseline: {np.linalg.norm(T)*1000:.1f} mm")
        return K1, D1, K2, D2, R, T
    except Exception as e:
        print(f"[Stereo] Could not load calibration: {e}")
        return None


def save_stereo_calibration(K1, D1, K2, D2, R, T):
    d = {
        "K1": K1.tolist(), "D1": D1.tolist(),
        "K2": K2.tolist(), "D2": D2.tolist(),
        "R":  R.tolist(),  "T":  T.tolist(),
    }
    with open(STEREO_CALIB_FILE, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[Stereo] Calibration saved → {STEREO_CALIB_FILE}")
    print(f"  Baseline: {np.linalg.norm(T)*1000:.1f} mm")


# ─────────────────────────────────────────────────────────────────────────────
# STEREO CALIBRATION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def run_stereo_calibration(pipe1, pipe2, K1, D1, K2, D2, detector):
    """
    Extrinsic-only stereo calibration using Tag 0 as the reference target.

    Intrinsics (K1, D1, K2, D2) are taken directly from the RealSense firmware
    — they are factory-calibrated and accurate enough for triangulation.

    Method
    ------
    Both cameras independently run PnP on Tag 0 to find where the tag sits in
    each camera's frame.  The relative camera-to-camera transform is then:

        R_12 = R2 @ R1.T        (rotation:    cam1 frame → cam2 frame)
        T_12 = t2 - R_12 @ t1  (translation: cam1 origin seen from cam2)

    CALIB_FRAMES estimates are collected and the median rotation (via Rodrigues)
    and median translation are saved for robustness against single-frame noise.

    Controls (during live preview)
    ------
    SPACE — capture the current frame pair (Tag 0 must be visible in both)
    ESC   — abort without saving

    Tips
    ----
    • Tag 0 must be clearly visible in BOTH cameras throughout collection.
    • Keep the tag stationary — move the collection window (press SPACE
      repeatedly) rather than moving the tag.
    • 30 frames at 10+ fps takes ~3 seconds; results are averaged automatically.
    """
    print(f"\n[Calib] Stereo calibration using Tag 0 as reference.")
    print(f"  Keep Tag 0 fully visible in BOTH cameras.")
    print(f"  Capturing {CALIB_FRAMES} frames automatically — ESC to abort.\n")

    R_samples = []   # list of (3,3) rotation matrices  R_cam2_from_cam1
    T_samples = []   # list of (3,)  translation vectors T_cam2_from_cam1

    CAPTURE_INTERVAL_S = 0.10   # min seconds between auto-captures
    last_capture_t = 0.0

    win = "Stereo Calibration (Tag 0) — cam1 | cam2    ESC=abort"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 360)

    while len(R_samples) < CALIB_FRAMES:
        f1 = pipe1.wait_for_frames().get_color_frame()
        f2 = pipe2.wait_for_frames().get_color_frame()
        if not f1 or not f2:
            continue

        img1 = np.asanyarray(f1.get_data())
        img2 = np.asanyarray(f2.get_data())

        # PnP on Tag 0 from each camera independently
        tags1 = detect_tags_rgb(img1, K1, D1, detector)
        tags2 = detect_tags_rgb(img2, K2, D2, detector)

        found_both = (0 in tags1) and (0 in tags2)

        # Auto-capture when tag visible in both cameras and interval has passed
        now_t = time.perf_counter()
        if found_both and (now_t - last_capture_t) >= CAPTURE_INTERVAL_S:
            rvec1, tvec1, _ = tags1[0]
            rvec2, tvec2, _ = tags2[0]
            R1, _ = cv2.Rodrigues(rvec1)
            R2, _ = cv2.Rodrigues(rvec2)
            t1 = tvec1.flatten()
            t2 = tvec2.flatten()

            # Relative transform: cam1 → cam2
            R_12 = R2 @ R1.T
            T_12 = t2 - R_12 @ t1

            R_samples.append(R_12)
            T_samples.append(T_12)
            last_capture_t = now_t
            print(f"  Captured {len(R_samples)}/{CALIB_FRAMES}  "
                  f"baseline so far: {np.linalg.norm(T_12)*1000:.1f} mm")

        disp1 = img1.copy()
        disp2 = img2.copy()

        n = len(R_samples)
        status_col = (0, 220, 0) if found_both else (0, 80, 255)
        status_txt = (f"Tag 0 in BOTH — capturing {n}/{CALIB_FRAMES}"
                      if found_both else
                      f"cam1={'OK' if 0 in tags1 else 'NOT FOUND'}  "
                      f"cam2={'OK' if 0 in tags2 else 'NOT FOUND'} — waiting for tag")

        for disp, tags, K, D in ((disp1, tags1, K1, D1), (disp2, tags2, K2, D2)):
            if 0 in tags:
                rvec, tvec, _ = tags[0]
                cv2.drawFrameAxes(disp, K, D,
                                  rvec.reshape(3, 1), tvec.reshape(3, 1), 0.05)
            cv2.putText(disp, status_txt, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)
            # Progress bar
            bar_w = int(disp.shape[1] * n / CALIB_FRAMES)
            cv2.rectangle(disp, (0, disp.shape[0]-10), (bar_w, disp.shape[0]),
                          (0, 220, 0), -1)

        combined = np.hstack([
            cv2.resize(disp1, (640, 360)),
            cv2.resize(disp2, (640, 360)),
        ])
        cv2.imshow(win, combined)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("[Calib] Aborted — nothing saved.")
            cv2.destroyWindow(win)
            return

    cv2.destroyWindow(win)

    # Median rotation via Rodrigues vectors (robust to outliers)
    rvec_samples = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_samples])
    rvec_med     = np.median(rvec_samples, axis=0)
    R_final, _   = cv2.Rodrigues(rvec_med)
    T_final      = np.median(T_samples, axis=0)

    # Spread diagnostics — warn if samples are inconsistent
    rvec_std = np.std(rvec_samples, axis=0)
    T_std    = np.std(T_samples, axis=0)
    print(f"\n[Calib] Calibration complete.")
    print(f"  Baseline     : {np.linalg.norm(T_final)*1000:.1f} mm")
    print(f"  T std (mm)   : {T_std*1000}  "
          f"{'✓ consistent' if np.all(T_std*1000 < 2.0) else '⚠ high spread — retake'}")
    print(f"  R std (mrad) : {rvec_std*1000}  "
          f"{'✓ consistent' if np.all(rvec_std*1000 < 5.0) else '⚠ high spread — retake'}")

    save_stereo_calibration(K1, D1, K2, D2, R_final, T_final)


# ─────────────────────────────────────────────────────────────────────────────
# TRIANGULATION
# ─────────────────────────────────────────────────────────────────────────────

def build_projection_matrices(K1, K2, R, T):
    """
    Build 3×4 projection matrices for both cameras.

    Camera 1 is the reference (world = camera-1 frame).
    Camera 2 is expressed relative to camera 1 via R, T.

    Returns P1 (3×4), P2 (3×4).
    """
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])
    return P1, P2


def triangulate_point(pt1, pt2, P1, P2):
    """
    Triangulate a single 3D point from matched 2D pixel coordinates.

    Parameters
    ----------
    pt1, pt2 : (2,) pixel coordinates (float) in camera 1 and camera 2
    P1, P2   : 3×4 projection matrices

    Returns
    -------
    (3,) 3D point in camera-1 frame (metres)
    """
    pts4d = cv2.triangulatePoints(
        P1, P2,
        pt1.astype(float).reshape(2, 1),
        pt2.astype(float).reshape(2, 1),
    )
    return (pts4d[:3] / pts4d[3]).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MARKER DETECTION  (2-D only, no depth)
# ─────────────────────────────────────────────────────────────────────────────

def build_color_mask(hsv_blurred, hsv_low, hsv_high):
    mask = cv2.inRange(hsv_blurred, hsv_low, hsv_high)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_blob_centroid(mask):
    """
    Find the single best circular blob in a binary mask.

    Returns (cx, cy) pixel centroid as floats, or None if nothing found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_circ = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MARKER_AREA_MIN or area > MARKER_AREA_MAX:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1.0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
        if circularity < CIRCULARITY_MIN:
            continue
        if circularity > best_circ:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            best_circ = circularity
            best = (M["m10"] / M["m00"], M["m01"] / M["m00"])

    return best


def detect_marker_2d(hsv_blurred, hsv_low, hsv_high):
    """
    Detect a colour marker in a single camera frame.

    Returns (cx, cy) pixel centroid or None.
    Uses shadow V-channel fallback if primary range finds nothing.
    """
    mask = build_color_mask(hsv_blurred, hsv_low, hsv_high)
    result = find_blob_centroid(mask)
    if result is not None:
        return result

    if SHADOW_V_SLACK > 0:
        shadow_low  = hsv_low.copy();  shadow_low[2]  = max(0,   hsv_low[2]  - SHADOW_V_SLACK)
        shadow_high = hsv_high.copy(); shadow_high[2] = min(255, hsv_high[2] + SHADOW_V_SLACK)
        mask = build_color_mask(hsv_blurred, shadow_low, shadow_high)
        result = find_blob_centroid(mask)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# APRILTAG DETECTION  (RGB only, pure PnP — no depth override)
# ─────────────────────────────────────────────────────────────────────────────

def build_aruco_detector():
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant    = 3
    params.minMarkerPerimeterRate    = 0.01
    params.maxMarkerPerimeterRate    = 4.0
    params.polygonalApproxAccuracyRate    = 0.08
    params.perspectiveRemovePixelPerCell  = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    params.errorCorrectionRate       = 1.0
    params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize   = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy   = 0.1
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        params,
    )


def detect_tags_rgb(img, K, D, detector, display_img=None):
    """
    Detect all AprilTags visible in img using pure PnP (no depth sensor).

    If display_img is provided, detected tag edges and IDs are drawn on it.

    Returns
    -------
    dict: tag_id (int) → (rvec (3,), tvec (3,), centroid (2,)) in camera frame
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}

    if display_img is not None:
        cv2.aruco.drawDetectedMarkers(display_img, corners, ids)

    tag_sizes = {0: TAG_SIZE, 1: TAG_SIZE_TRACKING, 2: TAG_SIZE_TRACKING}
    results = {}
    for i, tag_id in enumerate(ids.flatten()):
        sz = tag_sizes.get(int(tag_id), TAG_SIZE_TRACKING)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], sz, K, D)
        centroid = corners[i][0].mean(axis=0)
        results[int(tag_id)] = (rvecs[0].flatten(), tvecs[0].flatten(), centroid)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HSV TUNER
# ─────────────────────────────────────────────────────────────────────────────

def hsv_tuner(pipe, K, D, color_name, initial_low, initial_high):
    """
    Interactive HSV threshold tuner using camera 1's live feed.
    Press 's' to save, 'q'/ESC to cancel.
    """
    WIN_CTRL   = f"HSV Tuner: {color_name} — Controls  (s=save  r=reset  q=quit)"
    WIN_MASK   = f"HSV Tuner: {color_name} — Mask"
    WIN_RESULT = f"HSV Tuner: {color_name} — Result"

    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 600, 320)

    def nothing(_): pass
    for label, val, mx in [
        ("H min", initial_low[0],  179), ("H max", initial_high[0], 179),
        ("S min", initial_low[1],  255), ("S max", initial_high[1], 255),
        ("V min", initial_low[2],  255), ("V max", initial_high[2], 255),
    ]:
        cv2.createTrackbar(label, WIN_CTRL, int(val), mx, nothing)

    result = None
    while True:
        frame = pipe.wait_for_frames().get_color_frame()
        if not frame:
            continue
        img = np.asanyarray(frame.get_data())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.GaussianBlur(hsv, (5, 5), 0)

        lo = np.array([cv2.getTrackbarPos(f"{c} min", WIN_CTRL) for c in "HSV"])
        hi = np.array([cv2.getTrackbarPos(f"{c} max", WIN_CTRL) for c in "HSV"])

        mask = build_color_mask(hsv_b, lo, hi)
        pt   = find_blob_centroid(mask)
        disp = cv2.bitwise_and(img, img, mask=mask)
        if pt:
            cv2.circle(disp, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), 2)

        cv2.imshow(WIN_MASK,   mask)
        cv2.imshow(WIN_RESULT, disp)

        info = np.zeros((320, 600, 3), dtype=np.uint8)
        cv2.putText(info, f"Low : H={lo[0]:3d} S={lo[1]:3d} V={lo[2]:3d}",
                    (10,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info, f"High: H={hi[0]:3d} S={hi[1]:3d} V={hi[2]:3d}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(info, f"Blob: {'detected' if pt else 'none'}",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 0) if pt else (0, 80, 255), 2)
        cv2.putText(info, "s=SAVE   r=RESET   q=QUIT",
                    (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.imshow(WIN_CTRL, info)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            result = (lo.copy(), hi.copy()); break
        elif key == ord("r"):
            for label, val in zip(["H min","H max","S min","S max","V min","V max"],
                                   [*initial_low, *initial_high]):
                cv2.setTrackbarPos(label, WIN_CTRL, int(val))
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(WIN_CTRL)
    cv2.destroyWindow(WIN_MASK)
    cv2.destroyWindow(WIN_RESULT)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SETUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def list_cameras():
    ctx = rs.context()
    devices = ctx.devices
    if not devices:
        print("No RealSense devices found.")
        return
    print(f"Found {len(devices)} device(s):")
    for d in devices:
        sn   = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        print(f"  Serial: {sn}   Name: {name}")


def start_pipeline(serial, width=1280, height=720, fps=30):
    """
    Start a RealSense pipeline for a specific camera, colour stream only.

    Returns (pipeline, intrinsics_object).
    """
    pipe   = rs.pipeline()
    cfg    = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipe.start(cfg)
    vsp     = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr    = vsp.get_intrinsics()
    K = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1       ],
    ])
    D = np.array(intr.coeffs)
    return pipe, K, D


def build_timestamped_csv_path(base_name):
    root, ext = os.path.splitext(base_name)
    return f"{root}_{int(time.time())}{ext or '.csv'}"


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND FRAME CAPTURE THREAD
# ─────────────────────────────────────────────────────────────────────────────

class CaptureThread:
    """
    Runs wait_for_frames() for both pipelines in a daemon thread so the main
    loop is never blocked by frame acquisition.  The main loop reads the latest
    frame pair non-blocking via get_frame().

    pause() / resume() hand pipeline control back to the main thread so that
    run_stereo_calibration() and hsv_tuner() can call wait_for_frames() directly
    without conflicting with this thread.
    """

    def __init__(self, pipe1, pipe2):
        self._pipe1      = pipe1
        self._pipe2      = pipe2
        self._q          = queue.Queue(maxsize=1)
        self._stop       = threading.Event()
        self._paused     = threading.Event()   # set while thread is paused
        self._do_run     = threading.Event()   # cleared to request pause
        self._do_run.set()
        self._thread     = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def get_frame(self):
        """Return (img1, img2) or None if no new frame is ready."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def pause(self):
        """Signal the thread to pause and block until it has stopped acquiring."""
        self._do_run.clear()
        self._paused.wait(timeout=0.5)

    def resume(self):
        """Hand pipeline control back to the capture thread."""
        self._paused.clear()
        self._do_run.set()

    def stop(self):
        self._stop.set()
        self._do_run.set()          # unblock if waiting in pause
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            if not self._do_run.is_set():
                self._paused.set()
                self._do_run.wait()
                self._paused.clear()
                continue
            try:
                f1 = self._pipe1.wait_for_frames(timeout_ms=100)
                f2 = self._pipe2.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                continue
            cf1 = f1.get_color_frame()
            cf2 = f2.get_color_frame()
            if not cf1 or not cf2:
                continue
            img1 = np.asanyarray(cf1.get_data()).copy()
            img2 = np.asanyarray(cf2.get_data()).copy()
            # Keep only the latest — drop stale frame if queue is full
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put((img1, img2))


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING THREAD  (detection + CSV + display annotation)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingThread:
    """
    Consumes raw frames from CaptureThread, runs all detection (ArUco + colour),
    writes CSV, annotates display frames, and pushes (combined_img, plot_data)
    to the main thread via get_latest() — non-blocking.

    Timing state, persistence caches, and frame counter all live here so the
    main thread never blocks on detection work.
    """

    _PERSIST_COLOR = 0.5
    _PERSIST_TAG   = 0.5

    def __init__(self, capturer, csv_writer, csv_file,
                 K1, D1, K2, D2, P1, P2, detector,
                 enabled_markers, write_interval, warmup_s):
        self._capturer    = capturer
        self._csv_writer  = csv_writer
        self._csv_file    = csv_file
        self._detector    = detector
        self._enabled     = enabled_markers
        self._wi          = write_interval
        self._warmup_s    = warmup_s

        self._K1, self._D1 = K1.copy(), D1.copy()
        self._K2, self._D2 = K2.copy(), D2.copy()
        self._P1, self._P2 = P1, P2
        self._calib_lock   = threading.Lock()

        self._warmup_end       = None
        self._next_write       = None
        self._rec_start        = None   # perf_counter time recording began
        self._rec_start_wall   = None   # time.time() at same moment
        self._frame_count      = 0

        self._last_good      = {}
        self._last_tag_world = {}

        self._result_q = queue.Queue(maxsize=1)
        self._stop     = threading.Event()
        self._do_run   = threading.Event(); self._do_run.set()
        self._paused   = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self):   self._thread.start()

    def get_latest(self):
        try:    return self._result_q.get_nowait()
        except queue.Empty: return None

    def pause(self):
        self._do_run.clear()
        self._paused.wait(timeout=1.0)

    def resume(self):
        self._paused.clear()
        self._do_run.set()

    def update_calibration(self, K1, D1, K2, D2, P1, P2):
        with self._calib_lock:
            self._K1, self._D1 = K1.copy(), D1.copy()
            self._K2, self._D2 = K2.copy(), D2.copy()
            self._P1, self._P2 = P1, P2

    def stop(self):
        self._stop.set(); self._do_run.set()
        self._thread.join(timeout=3.0)

    @property
    def frame_count(self): return self._frame_count

    # ── Worker ─────────────────────────────────────────────────────────────

    def _put(self, combined, plot_data):
        try:    self._result_q.get_nowait()
        except queue.Empty: pass
        self._result_q.put((combined, plot_data))

    def _run(self):
        while not self._stop.is_set():
            if not self._do_run.is_set():
                self._paused.set()
                self._do_run.wait()
                self._paused.clear()
                continue

            frame_pair = self._capturer.get_frame()
            if frame_pair is None:
                time.sleep(0.001)
                continue

            with self._calib_lock:
                K1, D1 = self._K1.copy(), self._D1.copy()
                K2, D2 = self._K2.copy(), self._D2.copy()
                P1, P2 = self._P1, self._P2

            img1, img2 = frame_pair
            disp1, disp2 = img1.copy(), img2.copy()

            # ── Timing ───────────────────────────────────────────────────
            now = time.perf_counter()
            if self._warmup_end is None:
                self._warmup_end = now + self._warmup_s
            in_warmup = now < self._warmup_end

            if self._next_write is None and not in_warmup:
                self._next_write     = now
                self._rec_start      = now
                self._rec_start_wall = time.time()

            slots_due = 0
            if not in_warmup and now >= self._next_write:
                slots_due        = int((now - self._next_write) // self._wi) + 1
                self._next_write += slots_due * self._wi
            do_write = slots_due > 0

            if not (do_write or in_warmup):
                combined = np.hstack([cv2.resize(disp1, (640, 360)),
                                      cv2.resize(disp2, (640, 360))])
                self._put(combined, None)
                continue

            # ── Tag detection ─────────────────────────────────────────────
            tags1 = detect_tags_rgb(img1, K1, D1, self._detector, display_img=disp1)

            T_base = rvec_base = rmat_inv = tag0_pos = None
            if 0 in tags1:
                rvec_base, tvec_base, _ = tags1[0]
                rmat_raw, _ = cv2.Rodrigues(rvec_base)
                rmat_w = rmat_raw @ R_YZ_TO_XY if USE_TAG0_YZ_TO_XY_REMAP else rmat_raw
                rmat_w = rmat_w @ R_CAMERA_FLIP
                rmat_inv = rmat_w.T
                tag0_pos = tvec_base
                T_base   = True
                cv2.drawFrameAxes(disp1, K1, D1,
                                  rvec_base.reshape(3,1), tvec_base.reshape(3,1), 0.05)

            tag1_world = tag2_world = midpoint_world = None
            for tid in (1, 2):
                if tid in tags1 and rmat_inv is not None:
                    _, tvec, _ = tags1[tid]
                    w = rmat_inv @ (tvec - tag0_pos)
                    if tid == 1: tag1_world = w
                    else:        tag2_world = w
                    self._last_tag_world[tid] = (w, now)
                else:
                    cached = self._last_tag_world.get(tid)
                    if cached and (now - cached[1]) <= self._PERSIST_TAG:
                        if tid == 1: tag1_world = cached[0]
                        else:        tag2_world = cached[0]

            if tag1_world is not None and tag2_world is not None:
                midpoint_world = (tag1_world + tag2_world) / 2

            # ── Stereo sanity check ───────────────────────────────────────
            stereo_check = {}
            if P1 is not None and P2 is not None:
                tags2c = detect_tags_rgb(img2, K2, D2, self._detector, display_img=disp2)
                for tid in (1, 2):
                    if tid not in tags1 or tid not in tags2c:
                        continue
                    _, pnp_tv, c1 = tags1[tid]
                    _, _,      c2 = tags2c[tid]
                    sp = triangulate_point(np.array(c1), np.array(c2), P1, P2)
                    err = float(np.linalg.norm(sp - pnp_tv)) * 1000
                    sw  = (rmat_inv @ (sp - tag0_pos)) if rmat_inv is not None else None
                    stereo_check[tid] = (sw, err)

            # ── Colour markers ────────────────────────────────────────────
            hsv1 = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV), (5,5), 0)
            hsv2 = cv2.GaussianBlur(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV), (5,5), 0)
            marker_world = {}

            for name, cfg in MARKERS.items():
                if not cfg.get("enabled"):
                    continue
                pt1 = detect_marker_2d(hsv1, cfg["hsv_low"], cfg["hsv_high"])
                pt2 = detect_marker_2d(hsv2, cfg["hsv_low"], cfg["hsv_high"])
                if pt1 and pt2:
                    self._last_good[name] = (pt1, pt2, now)
                elif pt1 is None or pt2 is None:
                    cached = self._last_good.get(name)
                    if cached and (now - cached[2]) <= self._PERSIST_COLOR:
                        pt1, pt2 = cached[0], cached[1]
                    else:
                        pt1 = pt2 = None
                is_fresh = (pt1 and pt2 and name in self._last_good
                            and self._last_good[name][2] == now)
                bgr = cfg["bgr"]
                if T_base is not None:
                    for disp, pt in ((disp1, pt1), (disp2, pt2)):
                        if pt:
                            cv2.circle(disp, (int(pt[0]), int(pt[1])), 8, bgr,
                                       -1 if is_fresh else 2)
                if (P1 is not None and P2 is not None
                        and pt1 and pt2 and rmat_inv is not None):
                    p3 = triangulate_point(np.array(pt1), np.array(pt2), P1, P2)
                    marker_world[name] = rmat_inv @ (p3 - tag0_pos)

            # ── CSV write ─────────────────────────────────────────────────
            if do_write:
                bs = self._next_write - slots_due * self._wi
                for s in range(slots_due):
                    t_rel    = (bs + s * self._wi) - self._rec_start
                    abs_time = self._rec_start_wall + t_rel
                    row = [abs_time, t_rel]
                    row += ["0.000000","0.000000","0.000000"] if T_base else ["","",""]
                    for pos in (tag1_world, tag2_world, midpoint_world):
                        row += ([f"{pos[0]:.6f}",f"{pos[1]:.6f}",f"{pos[2]:.6f}"]
                                 if pos is not None else ["","",""])
                    for n in self._enabled:
                        pos = marker_world.get(n)
                        row += ([f"{pos[0]:.6f}",f"{pos[1]:.6f}",f"{pos[2]:.6f}"]
                                 if pos is not None else ["","",""])
                    self._csv_writer.writerow(row)
                    self._frame_count += 1
                self._csv_file.flush()

            # ── Display annotation ────────────────────────────────────────
            def _fmt(v):
                return (f"{v[0]*1000:.1f},{v[1]*1000:.1f},{v[2]*1000:.1f}mm"
                        if v is not None else "---")
            dy = 30
            for lbl, val in [("Tag1", tag1_world), ("Tag2", tag2_world),
                              ("Mid",  midpoint_world)]:
                cv2.putText(disp1, f"{lbl}: {_fmt(val)}",
                            (disp1.shape[1]-450, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                dy += 22
            for tid, (_, err) in sorted(stereo_check.items()):
                ok  = err < STEREO_CHECK_WARN_MM
                cv2.putText(disp1,
                            f"Tag{tid} PnP<>Stereo: {err:.0f}mm {'OK' if ok else 'WARN'}",
                            (disp1.shape[1]-450, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0,220,0) if ok else (0,80,255), 1)
                dy += 20
            for name, pos in marker_world.items():
                cv2.putText(disp1, f"{name}: {_fmt(pos)}", (20, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKERS[name]["bgr"], 1)
                dy += 22
            cv2.putText(disp1, "CALIBRATED" if P1 is not None else "NO CALIB (press c)",
                        (20, disp1.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,220,0) if P1 is not None else (0,80,255), 2)
            if in_warmup:
                rem = max(0.0, self._warmup_end - now)
                for d in (disp1, disp2):
                    cv2.putText(d, f"Starting in {rem:.1f}s",
                                (d.shape[1]//2-130, d.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 3)

            combined = np.hstack([cv2.resize(disp1,(640,360)),
                                   cv2.resize(disp2,(640,360))])
            plot_data = ({
                "do_write": do_write, "T_base": T_base is not None,
                "tag1_world": tag1_world, "tag2_world": tag2_world,
                "midpoint_world": midpoint_world, "stereo_check": stereo_check,
                "marker_world": marker_world, "P1_ok": P1 is not None,
            } if do_write else None)
            self._put(combined, plot_data)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Command-line flags ────────────────────────────────────────────────────
    if "--list-cameras" in sys.argv:
        list_cameras(); return

    load_marker_config()

    # ── Start both cameras ────────────────────────────────────────────────────
    print("Starting cameras …")
    try:
        pipe1, K1, D1 = start_pipeline(SERIAL_CAM1 or None, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
        print(f"  Camera 1 ready  serial={SERIAL_CAM1 or 'auto'}  {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")
    except Exception as e:
        print(f"  Camera 1 failed: {e}"); return

    try:
        pipe2, K2, D2 = start_pipeline(SERIAL_CAM2 or None, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
        print(f"  Camera 2 ready  serial={SERIAL_CAM2 or 'auto'}  {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")
    except Exception as e:
        print(f"  Camera 2 failed: {e}")
        pipe1.stop(); return

    # ── Load / run stereo calibration ─────────────────────────────────────────
    calib = load_stereo_calibration()
    if calib is None:
        print("[Stereo] No calibration found — press 'c' in the preview window "
              "to run stereo calibration, or place stereo_calibration.json in "
              f"{DIR}")
        P1 = P2 = None
    else:
        K1_cal, D1_cal, K2_cal, D2_cal, R_cal, T_cal = calib
        # Use calibrated intrinsics (more accurate than device-reported)
        K1, D1 = K1_cal, D1_cal
        K2, D2 = K2_cal, D2_cal
        P1, P2 = build_projection_matrices(K1, K2, R_cal, T_cal)

    detector = build_aruco_detector()

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_path = build_timestamped_csv_path(CSV_NAME)
    enabled_markers = [n for n, c in MARKERS.items() if c.get("enabled")]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Format Version", "1.23",
            "Take Name", os.path.splitext(os.path.basename(csv_path))[0],
            "Capture Start Time", now_str,
            "Capture Frame Rate", TARGET_FPS,
            "Export Frame Rate",  TARGET_FPS,
            "Length Units", "Meters",
        ])
        w.writerow([])
        rigid_cols = ["Tag0", "Tag1", "Tag2", "Midpoint"]
        w.writerow(["", "Type"]
                   + ["Rigid Body Marker"] * 3 * len(rigid_cols)
                   + ["Marker"]            * 3 * len(enabled_markers))
        name_row = ["", "Name"]
        for n in rigid_cols + enabled_markers:
            name_row += [n, n, n]
        w.writerow(name_row)
        w.writerow([])   # ID row (blank)
        sub_row = ["", ""]
        for _ in rigid_cols + enabled_markers:
            sub_row += ["Position", "Position", "Position"]
        w.writerow(sub_row)
        col_row = ["Timestamp (Unix s)", "Time Since Start (s)"]
        for _ in rigid_cols + enabled_markers:
            col_row += ["X", "Y", "Z"]
        w.writerow(col_row)

    tuner_select  = False
    last_combined = None

    WIN = "Triangulation — cam1 | cam2   (c=calibrate  t=HSV tuner  q=quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 360)

    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)

    capturer  = CaptureThread(pipe1, pipe2)
    processor = ProcessingThread(
        capturer, csv_writer, csv_file,
        K1, D1, K2, D2, P1, P2, detector,
        enabled_markers,
        write_interval=1.0 / TARGET_FPS,
        warmup_s=WARMUP_S,
    )
    capturer.start()
    processor.start()

    # ── Matplotlib 3D debug plot ──────────────────────────────────────────────
    if ENABLE_PLOT:
        plt.ion()
        fig = plt.figure(figsize=(7, 7))
        fig.suptitle("Triangulation Debug — World Frame (Tag 0 = origin)")
        ax3d = fig.add_subplot(111, projection="3d")
        ax3d.set_xlabel("X (mm)")
        ax3d.set_ylabel("Y (mm)")
        ax3d.set_zlabel("Z (mm)")

    try:
        while True:
            # ── Keys polled at ~200 Hz — nothing else runs here ───────────────
            key = cv2.waitKey(5) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("c"):
                capturer.pause(); processor.pause()
                run_stereo_calibration(pipe1, pipe2, K1, D1, K2, D2, detector)
                calib = load_stereo_calibration()
                if calib:
                    K1, D1, K2, D2, R_cal, T_cal = calib
                    P1, P2 = build_projection_matrices(K1, K2, R_cal, T_cal)
                    processor.update_calibration(K1, D1, K2, D2, P1, P2)
                capturer.resume(); processor.resume()
                continue

            if key == ord("t"):
                tuner_select = True

            if tuner_select:
                names = list(MARKERS.keys())
                if key == ord("0") or key == 27:
                    tuner_select = False
                else:
                    choice = next((i for i, ch in enumerate("123456789")
                                   if key == ord(ch) and i < len(names)), None)
                    if choice is not None:
                        tuner_select = False
                        sel = names[choice]
                        cfg = MARKERS[sel]
                        capturer.pause(); processor.pause()
                        res = hsv_tuner(pipe1, K1, D1, sel,
                                        cfg["hsv_low"].copy(), cfg["hsv_high"].copy())
                        capturer.resume(); processor.resume()
                        if res:
                            cfg["hsv_low"][:], cfg["hsv_high"][:] = res
                            save_marker_config()
                        continue

            # ── Get latest annotated frame from processing thread ─────────────
            result = processor.get_latest()
            if result is None:
                if tuner_select and last_combined is not None:
                    ov = last_combined.copy()
                    names = list(MARKERS.keys())
                    cv2.putText(ov, "HSV Tuner — press number:",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
                    for i, n in enumerate(names):
                        cv2.putText(ov, f"  {i+1}: {n}", (20, 65+i*28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,255), 2)
                    cv2.putText(ov, "  0: Cancel", (20, 65+len(names)*28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,80,255), 2)
                    cv2.imshow(WIN, ov)
                continue

            last_combined, plot_data = result

            if tuner_select:
                ov = last_combined.copy()
                names = list(MARKERS.keys())
                cv2.putText(ov, "HSV Tuner — press number:",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
                for i, n in enumerate(names):
                    cv2.putText(ov, f"  {i+1}: {n}", (20, 65+i*28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,255), 2)
                cv2.putText(ov, "  0: Cancel", (20, 65+len(names)*28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,80,255), 2)
                cv2.imshow(WIN, ov)
            else:
                cv2.imshow(WIN, last_combined)

            # ── Matplotlib — only on write frames, plot_data is None otherwise ─
            if ENABLE_PLOT and plot_data is not None:
                d = plot_data
                ax3d.clear()
                ax3d.scatter(0, 0, 0, color="black", s=120, marker="^",
                             label="Tag 0 (origin)", zorder=5)
                for tid, pos, clr in ((1, d["tag1_world"], "orange"),
                                      (2, d["tag2_world"], "magenta")):
                    if pos is not None:
                        p = pos * 1000
                        ax3d.scatter(p[0], p[1], p[2], color=clr, s=120,
                                     marker="D", label=f"Tag {tid} PnP", zorder=5)
                for tid, (sw, err) in sorted(d["stereo_check"].items()):
                    if sw is None: continue
                    p = sw * 1000
                    ok = err < STEREO_CHECK_WARN_MM
                    ax3d.scatter(p[0], p[1], p[2], color="lime" if ok else "red",
                                 s=60, marker="x", linewidths=2,
                                 label=f"Tag {tid} Stereo ({err:.0f}mm)")
                if d["midpoint_world"] is not None:
                    m = d["midpoint_world"] * 1000
                    ax3d.scatter(m[0], m[1], m[2], color="red", s=100,
                                 marker="o", label="Midpoint", zorder=5)
                    if d["tag1_world"] is not None and d["tag2_world"] is not None:
                        t1, t2 = d["tag1_world"]*1000, d["tag2_world"]*1000
                        ax3d.plot([t1[0],t2[0]], [t1[1],t2[1]], [t1[2],t2[2]],
                                  "m--", alpha=0.6, linewidth=1.5)
                for name, pos in d["marker_world"].items():
                    p = pos * 1000
                    bgr = MARKERS[name]["bgr"]
                    ax3d.scatter(p[0], p[1], p[2],
                                 color=(bgr[2]/255, bgr[1]/255, bgr[0]/255),
                                 s=80, marker="s", label=name, zorder=5)
                L = 200
                for vec, col, lbl in (((L,0,0),"red","X"),((0,L,0),"green","Y"),
                                       ((0,0,L),"blue","Z")):
                    ax3d.quiver(0,0,0,*vec, color=col, arrow_length_ratio=0.05,
                                linewidth=2, label=lbl)
                ax3d.set_xlim(-500,500); ax3d.set_ylim(-500,500); ax3d.set_zlim(-500,500)
                ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
                h, lb = ax3d.get_legend_handles_labels()
                ax3d.legend(dict(zip(lb,h)).values(), dict(zip(lb,h)).keys(),
                            loc="upper right", fontsize=7)
                tags_vis = (["0"] if d["T_base"] else []
                            + (["1"] if d["tag1_world"] is not None else [])
                            + (["2"] if d["tag2_world"] is not None else []))
                fig.suptitle(f"Triangulation Debug | Tags:{','.join(tags_vis) or 'none'}"
                             f" | Markers:{len(d['marker_world'])}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

    finally:
        processor.stop()
        capturer.stop()
        pipe1.stop()
        pipe2.stop()
        csv_file.close()
        if ENABLE_PLOT:
            plt.close("all")
        cv2.destroyAllWindows()
        print(f"\nRecording saved → {csv_path}  ({processor.frame_count} frames)")


if __name__ == "__main__":
    main()
