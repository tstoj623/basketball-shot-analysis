"""
Feature extraction from basketball shooting videos.
Uses MediaPipe for pose estimation and YOLO for person tracking.

UPDATED: 
- Uses ONLY pose_landmarks (normalized 3D) - NOT world landmarks
- Uses trained classifier for set point detection when available
- YOLO target-locking logic MATCHES the dataset generator exactly
- Side-view features added for side-view LLM analyzer
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Try to import set point classifier
try:
    from models.setpoint.set_point_model import (
        load_setpoint_model, 
        predict_setpoint_frame,
        predict_setpoint_with_confidence,
        is_model_loaded
    )
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("⚠️ Set point classifier not available, using heuristics only")


# ============================================================
# YOLO Configuration (MUST MATCH DATASET GENERATOR)
# ============================================================

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.25
IOU_KEEP_THRESHOLD = 0.25
BOX_MARGIN_SIDES = 0.25
BOX_MARGIN_TOP = 0.50
BOX_MARGIN_BOTTOM = 0.15
DETECT_EVERY = 2
HOLD_LAST_N = 12
SMOOTH_ALPHA = 0.80
MIN_CY_NORM = 0.0  # No vertical filtering



# ============================================================
# YOLO TARGET LOCKING HELPERS (same as dataset generator)
# ============================================================

def _clip_box_xyxy(box, W, H):
    """Clip box coordinates to frame boundaries."""
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, W - 1))
    y1 = int(np.clip(y1, 0, H - 1))
    x2 = int(np.clip(x2, 0, W - 1))
    y2 = int(np.clip(y2, 0, H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _box_center(box):
    """Get center point of a box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _norm_center_dist(cx, cy, W, H):
    """Normalized distance from frame center."""
    dx = (cx - W / 2) / (W / 2 + 1e-8)
    dy = (cy - H / 2) / (H / 2 + 1e-8)
    return float(np.sqrt(dx * dx + dy * dy))


def _iou(a, b):
    """Intersection over Union between two boxes."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)


def _ema_box(prev_smooth, new_box, alpha=0.75):
    """Exponential moving average for box smoothing."""
    if new_box is None:
        return prev_smooth
    nb = np.array(new_box, dtype=float)
    if prev_smooth is None:
        return tuple(nb.astype(int))
    pb = np.array(prev_smooth, dtype=float)
    sb = alpha * pb + (1 - alpha) * nb
    return tuple(sb.astype(int))


def yolo_person_candidates(result, W, H, class_id=0, conf_thres=0.25):
    """
    Get ALL person detection candidates from a YOLO result.
    Returns list of dicts with box, conf, dist, area, cy_norm.
    """
    if result is None or result.boxes is None:
        return []
    b = result.boxes
    if b.xyxy is None or b.cls is None:
        return []

    xyxy = b.xyxy.detach().cpu().numpy()
    cls = b.cls.detach().cpu().numpy().astype(int)
    conf = b.conf.detach().cpu().numpy() if getattr(b, "conf", None) is not None else np.ones(len(xyxy))

    out = []
    for bb, c, p in zip(xyxy, cls, conf):
        if c != class_id or float(p) < conf_thres:
            continue
        box = _clip_box_xyxy(tuple(bb.tolist()), W, H)
        if box is None:
            continue

        cx, cy = _box_center(box)
        dist = _norm_center_dist(cx, cy, W, H)
        area = ((box[2] - box[0]) * (box[3] - box[1])) / (W * H + 1e-8)
        cy_norm = cy / (H + 1e-8)

        out.append({
            "box": box,
            "conf": float(p),
            "dist": float(dist),
            "area": float(area),
            "cy_norm": float(cy_norm),
        })
    return out


def pick_initial_target(cands, w_area=0.40, w_center=0.50, w_conf=0.10,
                        min_cy_norm=0.0, w_low=0.30):
    """
    Pick initial target using blend of centrality, size, confidence, and vertical position.
    """
    if not cands:
        return None, None

    filtered = [r for r in cands if r.get("cy_norm", 0.0) >= float(min_cy_norm)]
    use = filtered if len(filtered) > 0 else cands

    scored = []
    for r in use:
        centrality = 1.0 - np.clip(r["dist"] / np.sqrt(2.0), 0.0, 1.0)
        score = (
            w_area * r["area"] +
            w_low * r.get("cy_norm", 0.0) +
            w_center * centrality +
            w_conf * r["conf"]
        )
        rr = dict(r)
        rr["score"] = float(score)
        scored.append(rr)

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[0]["box"], scored[0]


def locked_target_update(cands, locked_box, iou_keep=0.25, min_cy_norm=0.0):
    """
    TARGET LOCK: Update locked target based on IoU matching.
    Never switches to a different person unless explicitly allowed.
    """
    if locked_box is None:
        box, meta = pick_initial_target(cands, min_cy_norm=min_cy_norm)
        return box, meta, True if box is not None else False

    if not cands:
        return locked_box, None, False

    best_box, best_iou, best_meta = None, -1.0, None
    for r in cands:
        i = _iou(r["box"], locked_box)
        if i > best_iou:
            best_iou = i
            best_box = r["box"]
            best_meta = dict(r)
            best_meta["iou_to_locked"] = float(i)

    if best_box is not None and best_iou >= iou_keep:
        return best_box, best_meta, True

    return locked_box, best_meta, False



def expand_box_asymmetric(box, frame_w, frame_h, 
                          margin_sides=0.25, margin_top=0.50, margin_bottom=0.15):
    """
    Expand bounding box with ASYMMETRIC margins.
    More margin on top to capture raised arms during shooting.
    """
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    x1 = max(0, int(x1 - w * margin_sides))
    x2 = min(frame_w - 1, int(x2 + w * margin_sides))
    y1 = max(0, int(y1 - h * margin_top))
    y2 = min(frame_h - 1, int(y2 + h * margin_bottom))
    
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def angle2d(a, b, c):
    """Angle at point b given 2D points a,b,c in degrees."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = float(np.dot(ba, bc) / denom)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def angle3d(a, b, c):
    """Angle at point b in 3D space (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.clip(np.arccos(cosine_angle), 0, 180))


def calculate_distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def normalize_angle(angle):
    """Normalize angle to -180 to +180 range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def smooth_signal(arr, win=9, poly=2):
    """Savgol smoothing with fallback for short arrays."""
    arr = np.asarray(arr, float)
    if len(arr) >= win:
        if win % 2 == 0:
            win += 1
        return savgol_filter(arr, win, poly, mode="interp")
    return pd.Series(arr).rolling(window=min(5, len(arr)), center=True).mean().ffill().bfill().to_numpy()


def point_line_dist_2d(p, a, b):
    """Perpendicular distance from point p to line through a→b, all in 2D."""
    p, a, b = np.asarray(p[:2], float), np.asarray(a[:2], float), np.asarray(b[:2], float)
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-8:
        return float(np.linalg.norm(p - a))
    t = np.dot(p - a, ab) / ab_len_sq
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


# ============================================================
# BALL DETECTION UTILITY
# ============================================================

BALL_MODEL_PATH = Path(__file__).parent / "best (5).pt"


def get_ball_detections(video_path: str, conf: float = 0.30) -> dict:
    """
    Run ball detection model on a video.
    Returns {frame_idx: (cx_px, cy_px)} — highest-confidence detection per frame.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("   ⚠️  ultralytics not available, skipping ball detection")
        return {}

    if not BALL_MODEL_PATH.exists():
        print(f"   ⚠️  Ball model not found at {BALL_MODEL_PATH}, skipping")
        return {}

    model = YOLO(str(BALL_MODEL_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            best_i = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best_i].tolist()
            detections[frame_idx] = ((x1 + x2) / 2, (y1 + y2) / 2)
        frame_idx += 1

    cap.release()
    print(f"   🏀 Ball detections: {len(detections)}/{total} frames ({len(detections)/max(total,1):.0%} coverage)")
    return detections


# ============================================================
# PHASE DETECTION
# ============================================================

def detect_release_frame(df, K=5, ball_detections=None, use_classifier=True):
    """
    Detect release frame - the peak wrist height moment.

    When ball_detections is provided (dict of {frame_idx: (cx, cy)} from
    get_ball_detections), fuses the pose-based score with a ball-separation
    signal: the frame where ball-wrist distance transitions from its minimum
    to a sustained increase is the actual release moment.
    Final score = 0.6 * pose_score + 0.4 * ball_separation_signal.
    """
    if df is None or df.empty:
        return None

    if 'wrist_y_s' not in df.columns:
        print("   ❌ ERROR: wrist_y_s column missing!")
        return None
    
    wrist_y = df['wrist_y_s'].to_numpy(dtype=float)
    shoulder_y = df['shoulder_y_s'].to_numpy(dtype=float)
    elbow = df['elbow_s'].to_numpy(dtype=float)

    # Use absolute wrist height (inverted: lower pixel y = higher position)
    # shoulder_y - wrist_y was wrong for jumping shots — shoulder drops on landing
    # making the gap grow even after the wrist has peaked
    height = -wrist_y
    n = len(df)
    
    # Find ALL local maxima in height (= highest wrist positions)
    prom = max(1e-6, np.std(height) * 0.25)
    peaks, props = find_peaks(height, prominence=prom, distance=8)
    
    print(f"\n   📊 Release Detection:")
    print(f"      Wrist Y range: {wrist_y.min():.1f} to {wrist_y.max():.1f} ({wrist_y.max() - wrist_y.min():.1f} px)")
    print(f"      Found {len(peaks)} candidate peaks")

    # Discard peaks where wrist is still below shoulder — can't be a release
    above_shoulder = wrist_y[peaks] < shoulder_y[peaks]
    peaks_filtered = peaks[above_shoulder]
    if len(peaks_filtered) > 0:
        peaks = peaks_filtered
    print(f"      After height filter (wrist above shoulder): {len(peaks)} peaks")

    if len(peaks) == 0:
        # No prominence peak found — just use the frame where the wrist is highest
        best = int(np.argmax(height))
        print(f"      No valid peaks, using global max height at idx {best}")
        return best

    # Sort peaks by height (highest first)
    peaks_sorted = peaks[np.argsort(height[peaks])]
    top_peaks = peaks_sorted[-min(K, len(peaks_sorted)):]
    
    # Compute scoring factors
    vy = np.gradient(height)
    elbow_v = np.gradient(elbow)
    
    h_norm = (height - height.min()) / (height.max() - height.min() + 1e-8)
    ext = np.clip((elbow - 120.0) / (175.0 - 120.0), 0.0, 1.0)

    scores = np.zeros(n, dtype=float)

    for p in top_peaks:
        lo = max(0, p - 4)
        hi = min(n - 1, p + 4)
        idxs = np.arange(lo, hi + 1)
        elbow_stable = (elbow_v[idxs] > -2.0).astype(float)
        s = (0.70 * h_norm[idxs] + 0.30 * ext[idxs]) * elbow_stable
        scores[idxs] = np.maximum(scores[idxs], s)

    # Pick best directly from peak frames — don't smooth scores across peaks
    # (smoothing can elevate a secondary peak's neighbourhood over the true peak)
    peak_scores = [(p, scores[p]) for p in top_peaks]
    best_idx = max(peak_scores, key=lambda x: x[1])[0]

    frame_col = 'frame' if 'frame' in df.columns else 'Frame'
    best_frame = int(df.iloc[best_idx][frame_col])
    best_score = scores[best_idx]
    print(f"      Best release: idx {best_idx} (frame {best_frame}), score: {best_score:.3f}")

    # ── ML classifier override ───────────────────────────────────
    # To disable: pass use_classifier=False or delete release_model.joblib
    if use_classifier:
        try:
            from models.release.release_model import load_release_model, predict_release_frame, is_release_model_loaded
            if not is_release_model_loaded():
                load_release_model()
            if is_release_model_loaded():
                clf_idx, conf = predict_release_frame(df)
                if clf_idx is not None:
                    clf_frame = int(df.iloc[clf_idx][frame_col])
                    print(f"      Release classifier: idx {clf_idx} (frame {clf_frame}), conf={conf:.3f}")
                    best_idx = clf_idx
        except Exception as e:
            print(f"      ⚠️ Release classifier failed: {e}")
    # ────────────────────────────────────────────────────────────

    return best_idx


def detect_gather_frame(df, release_idx, search_window=60, K=5, fps=30.0, use_classifier=True):
    """
    Detect the gather/shot pocket using last valley + hip filter approach.
    Finds the last local minimum in wrist height before release,
    restricted to frames where wrist is at or above hip level.
    """
    if df is None or df.empty or release_idx is None:
        return None

    if 'wrist_y_s' not in df.columns or 'shoulder_y_s' not in df.columns:
        return None

    wrist_y = df['wrist_y_s'].to_numpy(dtype=float)
    shoulder_y = df['shoulder_y_s'].to_numpy(dtype=float)
    height = shoulder_y - wrist_y

    end_idx = release_idx
    start_idx = max(0, release_idx - search_window)
    if end_idx <= start_idx:
        return None

    search_height = height[start_idx:end_idx + 1]
    n_search = len(search_height)

    print(f"\n   📊 Gather Detection:")
    print(f"      Search window: idx {start_idx} to {end_idx}")

    if n_search < 5:
        return None

    vy = np.gradient(search_height)
    vy_smooth = savgol_filter(vy, min(7, len(vy) | 1), 2, mode="interp") if len(vy) >= 7 else vy

    # Zone score: hip-to-nose range gets highest score
    zone_score = np.ones(n_search)
    if 'hip_mid_y' in df.columns and 'r_wrist_y_px' in df.columns and 'nose_y' in df.columns:
        hip_px   = df['hip_mid_y'].to_numpy(dtype=float)[start_idx:end_idx + 1]
        nose_px  = df['nose_y'].to_numpy(dtype=float)[start_idx:end_idx + 1]
        wrist_px = df['r_wrist_y_px'].to_numpy(dtype=float)[start_idx:end_idx + 1]
        for i in range(n_search):
            if nose_px[i] <= wrist_px[i] <= hip_px[i]:
                zone_score[i] = 1.0
            elif wrist_px[i] < nose_px[i]:
                zone_score[i] = 0.15
            else:
                zone_score[i] = 0.05

    # Find valleys + zero crossings
    inv = -search_height
    prom = max(1e-6, np.std(inv) * 0.10)
    valleys, _ = find_peaks(inv, prominence=prom, distance=3)
    zc = [i for i in range(1, n_search - 1) if vy_smooth[i-1] <= 0 and vy_smooth[i+1] > 0]
    all_candidates = list(np.unique(np.concatenate([valleys, np.array(zc, dtype=int)])))
    if not all_candidates:
        all_candidates = list(range(n_search))

    # Score each candidate
    h_range = np.max(search_height) - np.min(search_height) + 1e-8
    elbow = df['elbow_s'].to_numpy(dtype=float)[start_idx:end_idx + 1]
    scored = []
    for c in all_candidates:
        h_sc = (np.max(search_height) - search_height[c]) / h_range
        elbow_sc = np.exp(-((elbow[c] - 90) ** 2) / (2 * 35 ** 2))
        total = 0.65 * zone_score[c] + 0.25 * h_sc + 0.10 * elbow_sc
        scored.append((c, total))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_local = scored[0][0]
    result_idx = start_idx + best_local

    # ── Low-valley + clean-rise override ────────────────────────────
    # 1. Filter valleys to lower 50% of wrist height range (kills setpoint candidates)
    # 2. Of those, find ones with clean uninterrupted rise to release
    # 3. Pick the LAST one that passes (most recent true gather)
    # To revert: delete this block
    DIP_THRESHOLD = 0.20
    h_min = np.min(search_height)
    h_max = np.max(search_height)
    h_midpoint = h_min + (h_max - h_min) * 0.50  # lower 50% threshold
    total_rise = max(search_height[-1] - h_min, 1e-6)

    low_candidates = [c for c in all_candidates if search_height[c] <= h_midpoint]

    clean_rise_idx = None
    for c in reversed(sorted(low_candidates)):  # last one first
        forward = search_height[c:]
        if len(forward) < 3:
            continue
        running_max = forward[0]
        passed = True
        for h in forward[1:]:
            running_max = max(running_max, h)
            if (running_max - h) > DIP_THRESHOLD * total_rise:
                passed = False
                break
        if passed:
            clean_rise_idx = start_idx + c
            break

    if clean_rise_idx is None and low_candidates:
        # No clean rise found — fall back to lowest low_candidate
        clean_rise_idx = start_idx + min(low_candidates, key=lambda c: search_height[c])

    if clean_rise_idx is not None and clean_rise_idx != result_idx:
        print(f"      Low+clean-rise gather: idx {clean_rise_idx} (zone scored: idx {result_idx})")
        result_idx = clean_rise_idx
    # ────────────────────────────────────────────────────────────────

    frame_col = 'frame' if 'frame' in df.columns else 'Frame'
    result_frame = int(df.iloc[result_idx][frame_col])
    print(f"      Best gather: idx {result_idx} (frame {result_frame})")

    # ── ML classifier override ───────────────────────────────────
    # To disable: pass use_classifier=False or delete this block
    if use_classifier:
        try:
            from models.gather.gather_model import load_gather_model, predict_gather_frame, is_gather_model_loaded
            if not is_gather_model_loaded():
                load_gather_model()
            if is_gather_model_loaded():
                clf_idx, conf = predict_gather_frame(df, release_idx, search_window)
                if clf_idx is not None:
                    clf_frame = int(df.iloc[clf_idx][frame_col])
                    print(f"      Gather classifier: idx {clf_idx} (frame {clf_frame}), conf={conf:.3f}")
                    result_idx = clf_idx
        except Exception as e:
            print(f"      ⚠️ Gather classifier failed: {e}")
    # ────────────────────────────────────────────────────────────

    return result_idx


def detect_setpoint_frame_heuristic(df, gather_idx, release_idx, K=5):
    """
    Set point detection: find when the wrist crosses face (nose) level while rising.

    The basketball set point is the moment the ball reaches the player's
    shooting window — traditionally at eye/nose level — before the final
    extension to release.  In pixel coords: wrist_y_s ≈ nose_y while vy < 0.

    Strategy (primary):
      1. In window [gather+2, release-2], find the frame where the wrist
         first crosses from below nose level to above nose level (upward).
      2. If no crossing exists, find the rising frame where |wrist - nose|
         is smallest.

    Fallback (when nose_y unavailable):
      Find the frame of minimum wrist height (maximum wrist elevation).
    """
    if df is None or df.empty or release_idx is None:
        return None

    if 'wrist_y_s' not in df.columns:
        mid = (gather_idx + release_idx) // 2 if gather_idx is not None else release_idx - 5
        return mid

    wrist_y = df['wrist_y_s'].to_numpy(dtype=float)

    # Search window: up to 1.5s before release, floored by gather
    window_start = max(0, release_idx - 45)
    window_end   = max(0, release_idx - 2)
    if gather_idx is not None:
        window_start = max(window_start, gather_idx + 2)

    if window_end - window_start < 3:
        return (window_start + window_end) // 2

    seg_wrist = wrist_y[window_start:window_end + 1]
    n = len(seg_wrist)

    vy_raw = np.gradient(seg_wrist)
    vy     = savgol_filter(vy_raw, min(5, n | 1), 2, mode="interp") if n >= 5 else vy_raw

    # ── Primary: nose-level crossing ─────────────────────────────────
    if 'nose_y' in df.columns:
        nose_y   = df['nose_y'].to_numpy(dtype=float)
        seg_nose = nose_y[window_start:window_end + 1]

        # diff > 0 means wrist is below nose (higher pixel y), < 0 means above
        diff = seg_wrist - seg_nose

        # Find first upward crossing (diff goes from positive to negative)
        crossing = None
        for i in range(1, n):
            if diff[i - 1] > 0 and diff[i] <= 0:
                # Pick whichever side is closer to zero
                crossing = i - 1 if abs(diff[i - 1]) <= abs(diff[i]) else i
                break

        if crossing is not None:
            best_local = crossing
        else:
            # No clean crossing — find rising frame closest to nose level
            rising = vy < 0
            abs_diff = np.abs(diff)
            if rising.any():
                masked = np.where(rising, abs_diff, np.inf)
                best_local = int(np.argmin(masked))
            else:
                best_local = int(np.argmin(abs_diff))

        best_abs = window_start + best_local
        gap = release_idx - best_abs
        wrist_val = seg_wrist[best_local]
        nose_val  = seg_nose[best_local]
        print(f"      [nose]     setpoint idx {best_abs}: wrist_y={wrist_val:.0f}, "
              f"nose_y={nose_val:.0f}, diff={wrist_val-nose_val:+.0f}px, gap={gap}f")

    else:
        best_abs = None

    # ── Midpoint approach (experimental comparison) ───────────────────
    # Set point = frame where wrist first reaches 50% of the way between
    # the shot pocket (wrist lowest = max pixel y) and peak extension
    # (wrist highest = min pixel y). Player-agnostic: no height assumption.
    shot_pocket_y = float(np.max(seg_wrist))   # wrist at lowest (shot pocket)
    peak_y        = float(np.min(seg_wrist))   # wrist at highest (full extension)
    target_y      = (shot_pocket_y + peak_y) / 2.0

    # First frame where wrist rises through the midpoint
    mid_local = None
    for i in range(n - 1):
        if seg_wrist[i] >= target_y and seg_wrist[i + 1] < target_y:
            mid_local = i if abs(seg_wrist[i] - target_y) <= abs(seg_wrist[i + 1] - target_y) else i + 1
            break
    if mid_local is None:
        mid_local = int(np.argmin(np.abs(seg_wrist - target_y)))

    mid_abs = window_start + mid_local
    mid_gap = release_idx - mid_abs
    print(f"      [midpoint] setpoint idx {mid_abs}: wrist_y={seg_wrist[mid_local]:.0f}, "
          f"target={target_y:.0f} (pocket={shot_pocket_y:.0f}, peak={peak_y:.0f}), gap={mid_gap}f")

    # ── Return nose result (primary), fall back to midpoint ───────────
    if best_abs is not None:
        return best_abs

    # Fallback: maximum wrist height in window
    best_local = int(np.argmin(seg_wrist))
    best_abs   = window_start + best_local
    gap = release_idx - best_abs
    print(f"      [max-height] setpoint idx {best_abs} (fallback, gap={gap}f)")
    return best_abs


def detect_all_phases(df, fps=30.0, use_classifier=True):
    """
    Detect all three shot phases: gather, set point, and release.
    """
    print("\n" + "="*60)
    print("   PHASE DETECTION")
    print("="*60)
    
    required_cols = ['wrist_y_s', 'elbow_s', 'frame']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"   ❌ Missing required columns: {missing}")
        return {
            'gather_frame': None, 'setpoint_frame': None, 'release_frame': None,
            'gather_idx': None, 'setpoint_idx': None, 'release_idx': None,
            'phase_durations': {}, 'setpoint_method': 'none'
        }
    
    print(f"   DataFrame: {len(df)} rows")
    
    release_idx = detect_release_frame(df, use_classifier=use_classifier)
    
    if release_idx is None:
        return {
            'gather_frame': None, 'setpoint_frame': None, 'release_frame': None,
            'gather_idx': None, 'setpoint_idx': None, 'release_idx': None,
            'phase_durations': {}, 'setpoint_method': 'none'
        }
    
    gather_idx = detect_gather_frame(df, release_idx, search_window=int(fps * 2), fps=fps, use_classifier=use_classifier)
    
    setpoint_idx = None
    setpoint_method = 'none'
    setpoint_confidence = 0.0
    
    # Always compute heuristic — used as fallback and for blending.
    print(f"\n   📊 Set Point Detection (Heuristic):")
    heuristic_idx = detect_setpoint_frame_heuristic(df, gather_idx, release_idx)
    if heuristic_idx is not None:
        frame_col = 'frame' if 'frame' in df.columns else 'Frame'
        print(f"      Heuristic setpoint: idx {heuristic_idx} (frame {int(df.iloc[heuristic_idx][frame_col])})")

    CONFIDENCE_HIGH = 0.70   # use classifier outright
    CONFIDENCE_LOW  = 0.45   # use heuristic outright; blend in between

    if use_classifier and CLASSIFIER_AVAILABLE:
        print(f"\n   📊 Set Point Detection (Classifier):")

        if not is_model_loaded():
            load_setpoint_model()

        if is_model_loaded():
            result = predict_setpoint_with_confidence(df, gather_idx, release_idx)

            if result['idx'] is not None:
                clf_idx = result['idx']
                setpoint_confidence = result['confidence']
                print(f"      Classifier prediction: idx {clf_idx} (frame {result['frame']})")
                print(f"      Confidence: {setpoint_confidence:.3f}")

                if setpoint_confidence >= CONFIDENCE_HIGH:
                    # High confidence — trust classifier
                    setpoint_idx = clf_idx
                    setpoint_method = 'classifier'
                    print(f"      ✅ Using classifier (high confidence)")
                elif setpoint_confidence <= CONFIDENCE_LOW or heuristic_idx is None:
                    # Low confidence — use heuristic
                    setpoint_idx = heuristic_idx
                    setpoint_method = 'heuristic'
                    print(f"      ⚠️  Low confidence — using heuristic")
                else:
                    # Medium confidence — blend weighted by confidence
                    w_clf = setpoint_confidence
                    w_heu = 1.0 - setpoint_confidence
                    blended = int(round(w_clf * clf_idx + w_heu * heuristic_idx))
                    # Clamp to valid window
                    blended = max(gather_idx, min(release_idx - 1, blended))
                    setpoint_idx = blended
                    setpoint_method = 'blended'
                    print(f"      🔀 Blended: clf={clf_idx} ({w_clf:.2f}) + heu={heuristic_idx} ({w_heu:.2f}) → {blended}")
            else:
                print(f"      ⚠️ Classifier failed, using heuristic")

    if setpoint_idx is None:
        setpoint_idx = heuristic_idx
        setpoint_method = 'heuristic'
    
    frame_col = 'frame' if 'frame' in df.columns else 'Frame'
    release_frame = int(df.iloc[release_idx][frame_col]) if release_idx is not None else None
    gather_frame = int(df.iloc[gather_idx][frame_col]) if gather_idx is not None else None
    setpoint_frame = int(df.iloc[setpoint_idx][frame_col]) if setpoint_idx is not None else None
    
    phase_durations = {}
    if gather_frame is not None and release_frame is not None:
        total_frames = release_frame - gather_frame
        phase_durations['total_frames'] = total_frames
        phase_durations['total_sec'] = total_frames / fps if fps > 0 else None
        
        if setpoint_frame is not None:
            gather_to_set = setpoint_frame - gather_frame
            set_to_release = release_frame - setpoint_frame
            phase_durations['gather_to_setpoint_frames'] = gather_to_set
            phase_durations['setpoint_to_release_frames'] = set_to_release
    
    print("\n" + "="*60)
    print(f"   FINAL PHASES:")
    print(f"      Gather:    frame {gather_frame} (idx {gather_idx})")
    print(f"      Set Point: frame {setpoint_frame} (idx {setpoint_idx}) [{setpoint_method}]")
    print(f"      Release:   frame {release_frame} (idx {release_idx})")
    print("="*60)
    
    return {
        'gather_frame': gather_frame,
        'setpoint_frame': setpoint_frame,
        'release_frame': release_frame,
        'gather_idx': gather_idx,
        'setpoint_idx': setpoint_idx,
        'release_idx': release_idx,
        'phase_durations': phase_durations,
        'setpoint_method': setpoint_method,
        'setpoint_confidence': setpoint_confidence
    }


# ============================================================
# PHASE-DEPENDENT SIDE-VIEW FEATURES
# ============================================================

def compute_phase_dependent_features(df, phases, fps=30.0):
    """
    Compute features that require phase info (drift baselines, timing, etc.).
    Call this AFTER detect_all_phases().
    Adds columns in-place to df.
    """
    gather_idx = phases.get('gather_idx')
    setpoint_idx = phases.get('setpoint_idx')
    release_idx = phases.get('release_idx')

    # --- Hip / Shoulder drift from gather baseline ---
    if gather_idx is not None and 'hip_mid_x' in df.columns:
        hip_base = df.loc[df.index[gather_idx], 'hip_mid_x']
        shoulder_base = df.loc[df.index[gather_idx], 'shoulder_mid_x']
        sw = df['shoulder_width']
        df['hip_drift_x_norm'] = (df['hip_mid_x'] - hip_base) / sw
        df['shoulder_drift_x_norm'] = (df['shoulder_mid_x'] - shoulder_base) / sw
    else:
        df['hip_drift_x_norm'] = 0.0
        df['shoulder_drift_x_norm'] = 0.0

    # --- Torso lean change from gather baseline ---
    if gather_idx is not None and 'torso_lean_deg' in df.columns:
        lean_base = df.loc[df.index[gather_idx], 'torso_lean_deg']
        df['torso_lean_change_deg'] = df['torso_lean_deg'] - lean_base
    else:
        df['torso_lean_change_deg'] = 0.0

    # --- Rolling min wrist-torso distance (gather → setpoint window) ---
    if 'wrist_torso_dist_norm' in df.columns:
        df['min_wrist_torso_dist_norm'] = (
            df['wrist_torso_dist_norm']
            .rolling(window=5, center=True, min_periods=1)
            .min()
        )
    else:
        df['min_wrist_torso_dist_norm'] = 0.0

    # --- Release separation (guide hand value at release) ---
    if release_idx is not None and 'hand_sep_norm_smooth' in df.columns:
        release_sep = df.loc[df.index[release_idx], 'hand_sep_norm_smooth']
        df['release_sep_norm'] = release_sep  # constant column for reference
    else:
        df['release_sep_norm'] = 0.0

    # --- Knee extension timing ---
    # Measures frame offset between peak knee bend and start of wrist rise
    if (gather_idx is not None and release_idx is not None 
            and 'knee_angle_right' in df.columns and 'wrist_vy' in df.columns):
        g, r = gather_idx, release_idx
        knee_seg = df['knee_angle_right'].iloc[g:r+1].to_numpy(dtype=float)
        wrist_vy_seg = df['wrist_vy'].iloc[g:r+1].to_numpy(dtype=float)
        
        if len(knee_seg) > 3:
            # Peak knee bend = min knee angle in segment
            peak_bend_local = int(np.argmin(knee_seg))
            
            # Start of wrist rise = first frame where vy is strongly negative (rising)
            vy_threshold = -np.std(np.abs(wrist_vy_seg)) * 0.3
            rise_start_local = None
            for i in range(len(wrist_vy_seg)):
                if wrist_vy_seg[i] < vy_threshold:
                    rise_start_local = i
                    break
            
            if rise_start_local is not None:
                timing = peak_bend_local - rise_start_local  # positive = knees lead arms (good)
                df['knee_extension_timing'] = float(timing)
            else:
                df['knee_extension_timing'] = 0.0
        else:
            df['knee_extension_timing'] = 0.0
    else:
        df['knee_extension_timing'] = 0.0

    # --- Hip-wrist velocity lag (kinetic chain timing) ---
    # Positive = hip peaks before wrist (good chain), negative = arms fire first (bad)
    if (gather_idx is not None and release_idx is not None
            and 'hip_vy' in df.columns and 'wrist_vy' in df.columns):
        g, r = gather_idx, release_idx
        hip_vy_seg = df['hip_vy'].iloc[g:r+1].to_numpy(dtype=float)
        wrist_vy_seg = df['wrist_vy'].iloc[g:r+1].to_numpy(dtype=float)
        if len(hip_vy_seg) > 5:
            hip_peak_local = int(np.argmin(hip_vy_seg))     # most negative = fastest upward
            wrist_peak_local = int(np.argmin(wrist_vy_seg))
            df['hip_wrist_vel_lag'] = float(wrist_peak_local - hip_peak_local)
        else:
            df['hip_wrist_vel_lag'] = 0.0
    else:
        df['hip_wrist_vel_lag'] = 0.0

    # --- Knee-wrist velocity lag (leg drive → arm fire) ---
    if (gather_idx is not None and release_idx is not None
            and 'knee_pos_vy' in df.columns and 'wrist_vy' in df.columns):
        g, r = gather_idx, release_idx
        knee_vy_seg = df['knee_pos_vy'].iloc[g:r+1].to_numpy(dtype=float)
        wrist_vy_seg = df['wrist_vy'].iloc[g:r+1].to_numpy(dtype=float)
        if len(knee_vy_seg) > 5:
            knee_peak_local = int(np.argmin(knee_vy_seg))
            wrist_peak_local = int(np.argmin(wrist_vy_seg))
            df['knee_wrist_vel_lag'] = float(wrist_peak_local - knee_peak_local)
        else:
            df['knee_wrist_vel_lag'] = 0.0
    else:
        df['knee_wrist_vel_lag'] = 0.0

    # --- Phase gap frames (lower body done → upper body starts) ---
    # Frames between knee extension start and wrist rise start
    # (similar to knee_extension_timing but specifically measuring the gap)
    if (gather_idx is not None and release_idx is not None
            and 'knee_vy' in df.columns and 'wrist_vy' in df.columns):
        g, r = gather_idx, release_idx
        knee_vy_seg = df['knee_vy'].iloc[g:r+1].to_numpy(dtype=float)
        wrist_vy_seg = df['wrist_vy'].iloc[g:r+1].to_numpy(dtype=float)
        
        if len(knee_vy_seg) > 5:
            # Knee extension start: first frame knee_vy goes positive (extending)
            knee_ext_start = None
            for i in range(len(knee_vy_seg)):
                if knee_vy_seg[i] > np.std(np.abs(knee_vy_seg)) * 0.2:
                    knee_ext_start = i
                    break
            
            # Wrist rise start: first frame wrist_vy strongly negative
            wrist_rise_start = None
            for i in range(len(wrist_vy_seg)):
                if wrist_vy_seg[i] < -np.std(np.abs(wrist_vy_seg)) * 0.3:
                    wrist_rise_start = i
                    break
            
            if knee_ext_start is not None and wrist_rise_start is not None:
                gap = wrist_rise_start - knee_ext_start
                df['phase_gap_frames'] = float(max(0, gap))
            else:
                df['phase_gap_frames'] = 0.0
        else:
            df['phase_gap_frames'] = 0.0
    else:
        df['phase_gap_frames'] = 0.0

    return df


# ============================================================
# MAIN FEATURE EXTRACTION (using ONLY pose_landmarks - NOT world)
# ============================================================

def extract_all_features(video_path, use_yolo=True, swap_hands=False):
    """
    Extract ALL frame-by-frame features from a video.
    Uses ONLY pose_landmarks (normalized 3D) - NOT world landmarks.
    Uses YOLO target-locking that MATCHES the dataset generator.
    Returns a DataFrame with one row per frame.
    
    Includes both front-view and side-view features.
    """
    print(f"\n🎬 Processing: {Path(video_path).name}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ Could not open video")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   📹 FPS: {fps:.1f}, Frames: {total_frames}, Res: {W}x{H}")

    # Initialize YOLO with target locking (MATCHES DATASET GENERATOR)
    yolo = None
    if use_yolo:
        try:
            from ultralytics import YOLO
            yolo = YOLO(YOLO_MODEL_PATH)
            print(f"   🎯 YOLO loaded with target locking (conf={YOLO_CONF_THRESHOLD}, iou_keep={IOU_KEEP_THRESHOLD})")
        except ImportError:
            print("   ⚠️ YOLO not available, using full frame")
            use_yolo = False

    # PASS 1: Read all frames and run YOLO with target locking
    frames = []
    boxes_by_frame = {}

    locked_box = None
    smooth_box = None
    lost_frames = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        Hf, Wf = frame.shape[:2]

        if yolo is not None:
            if frame_idx % DETECT_EVERY == 0:
                results = yolo.predict(frame, verbose=False)
                r0 = results[0] if results and len(results) > 0 else None
                cands = yolo_person_candidates(r0, Wf, Hf, conf_thres=YOLO_CONF_THRESHOLD)

                locked_box, meta, matched = locked_target_update(
                    cands, locked_box, iou_keep=IOU_KEEP_THRESHOLD, min_cy_norm=MIN_CY_NORM
                )

                if matched:
                    lost_frames = 0
                else:
                    lost_frames += 1
            else:
                lost_frames += 1

            use_box = locked_box if (locked_box is not None and lost_frames <= HOLD_LAST_N) else None
            smooth_box = _ema_box(smooth_box, use_box, alpha=SMOOTH_ALPHA)
            boxes_by_frame[frame_idx] = smooth_box
        else:
            boxes_by_frame[frame_idx] = None

        frame_idx += 1

    cap.release()
    print(f"   📦 Collected {len(frames)} frames")

    if len(frames) == 0:
        pose.close()
        return None

    # PASS 2: Extract pose from each frame using locked boxes
    frame_features_list = []

    for idx, frame in enumerate(frames):
        h, w = frame.shape[:2]
        
        if use_yolo and boxes_by_frame.get(idx) is not None:
            box = expand_box_asymmetric(
                boxes_by_frame[idx], w, h,
                margin_sides=BOX_MARGIN_SIDES,
                margin_top=BOX_MARGIN_TOP,
                margin_bottom=BOX_MARGIN_BOTTOM
            )
            if box is None:
                box = (0, 0, w, h)
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
            x1, y1 = 0, 0
            x2, y2 = w, h

        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(crop_rgb)

        if not pose_results.pose_landmarks:
            continue
        
        lm = pose_results.pose_landmarks.landmark
        mp_lm = mp_pose.PoseLandmark
        
        crop_h, crop_w = crop.shape[:2]

        def get_joint_3d(landmark_id):
            """Get joint position: x,y in full-frame pixels, z as normalized depth."""
            lmk = lm[landmark_id.value]
            px = x1 + lmk.x * crop_w
            py = y1 + lmk.y * crop_h
            pz = lmk.z * crop_w
            return np.array([px, py, pz])

        def get_visibility(landmark_id):
            """Get landmark visibility score (0-1)."""
            return float(lm[landmark_id.value].visibility)

        # Extract all joint positions
        joints = {
            'left_shoulder': get_joint_3d(mp_lm.LEFT_SHOULDER),
            'right_shoulder': get_joint_3d(mp_lm.RIGHT_SHOULDER),
            'left_elbow': get_joint_3d(mp_lm.LEFT_ELBOW),
            'right_elbow': get_joint_3d(mp_lm.RIGHT_ELBOW),
            'left_wrist': get_joint_3d(mp_lm.LEFT_WRIST),
            'right_wrist': get_joint_3d(mp_lm.RIGHT_WRIST),
            'left_hip': get_joint_3d(mp_lm.LEFT_HIP),
            'right_hip': get_joint_3d(mp_lm.RIGHT_HIP),
            'left_knee': get_joint_3d(mp_lm.LEFT_KNEE),
            'right_knee': get_joint_3d(mp_lm.RIGHT_KNEE),
            'left_ankle': get_joint_3d(mp_lm.LEFT_ANKLE),
            'right_ankle': get_joint_3d(mp_lm.RIGHT_ANKLE),
            'nose': get_joint_3d(mp_lm.NOSE),
            'left_thumb': get_joint_3d(mp_lm.LEFT_THUMB),
            'left_index': get_joint_3d(mp_lm.LEFT_INDEX),
            'left_pinky': get_joint_3d(mp_lm.LEFT_PINKY),
            'left_heel': get_joint_3d(mp_lm.LEFT_HEEL),
            'right_heel': get_joint_3d(mp_lm.RIGHT_HEEL),
            'left_foot_index': get_joint_3d(mp_lm.LEFT_FOOT_INDEX),
            'right_foot_index': get_joint_3d(mp_lm.RIGHT_FOOT_INDEX),
        }

        # Swap left/right joints for flipped lefty videos
        if swap_hands:
            swap_pairs = [
                ('left_shoulder', 'right_shoulder'),
                ('left_elbow', 'right_elbow'),
                ('left_wrist', 'right_wrist'),
                ('left_hip', 'right_hip'),
                ('left_knee', 'right_knee'),
                ('left_ankle', 'right_ankle'),
                ('left_heel', 'right_heel'),
                ('left_foot_index', 'right_foot_index'),
            ]
            for l, r in swap_pairs:
                joints[l], joints[r] = joints[r].copy(), joints[l].copy()
        
        # Pixel coordinates for phase detection
        r_wrist_px = joints['right_wrist'][:2]
        r_elbow_px = joints['right_elbow'][:2]
        r_shoulder_px = joints['right_shoulder'][:2]
        
        r_elbow_angle_2d = angle2d(r_shoulder_px, r_elbow_px, r_wrist_px)

        # Calculate normalization factors
        shoulder_mid = 0.5 * (joints['left_shoulder'] + joints['right_shoulder'])
        hip_mid = 0.5 * (joints['left_hip'] + joints['right_hip'])
        ankle_mid = 0.5 * (joints['left_ankle'] + joints['right_ankle'])
        knee_mid = 0.5 * (joints['left_knee'] + joints['right_knee'])
        shoulder_width = calculate_distance(joints['left_shoulder'][:2], joints['right_shoulder'][:2])
        torso_len = calculate_distance(shoulder_mid[:2], hip_mid[:2])
        shoulder_width = max(shoulder_width, 1e-6)
        torso_len = max(torso_len, 1e-6)

        # ============================================================
        # FRONT-VIEW FEATURES (existing)
        # ============================================================

        frame_features = {
            'shot': Path(video_path).name,
            'frame': idx,
            'Frame': idx,

            # PIXEL COORDINATES FOR PHASE DETECTION
            'r_wrist_y_px': r_wrist_px[1],
            'r_wrist_x_px': r_wrist_px[0],
            'r_elbow_y_px': r_elbow_px[1],
            'r_shoulder_y_px': r_shoulder_px[1],
            'r_elbow_angle': r_elbow_angle_2d,

            # ELBOW FLARE FEATURES
            'elbow_flare': abs(joints['right_elbow'][0] - joints['right_shoulder'][0]) / shoulder_width,
            'elbow_from_midline': abs(joints['right_elbow'][0] - shoulder_mid[0]) / shoulder_width,
            'shooting_plane_deviation': abs(joints['right_wrist'][0] - joints['right_elbow'][0]) / shoulder_width,
            'wrist_from_midline': (joints['right_wrist'][0] - shoulder_mid[0]) / shoulder_width,
            'elbow_from_midline_signed': (joints['right_elbow'][0] - shoulder_mid[0]) / shoulder_width,
            'elbow_wrist_offset': (joints['right_elbow'][0] - joints['right_wrist'][0]) / shoulder_width,
            'elbow_angle_frontal': 0.0,

            # ANGLES
            'elbow_angle_3d': angle3d(joints['right_shoulder'], joints['right_elbow'], joints['right_wrist']),

            # JOINT POSITIONS (normalized MediaPipe coords, 0-1)
            'right_wrist_y': joints['right_wrist'][1],
            'right_wrist_x': joints['right_wrist'][0],
            'right_elbow_y': joints['right_elbow'][1],
            'right_shoulder_y': joints['right_shoulder'][1],

            # TWIST FEATURES
            'hip_mid_x': hip_mid[0],
            'hip_mid_y': hip_mid[1],
            'shoulder_mid_x': shoulder_mid[0],
            'shoulder_mid_y': shoulder_mid[1],
            'stance_angle': 0.0,
            'hip_absolute_yaw': 0.0,
            'shoulder_absolute_yaw': 0.0,
            'hip_twist_from_feet': 0.0,
            'shoulder_twist_from_feet': 0.0,
            'twist_yaw_deg': 0.0,

            # GUIDE HAND FEATURES
            'left_wrist_y': joints['left_wrist'][1],
            'left_wrist_x': joints['left_wrist'][0],
            'hand_separation': calculate_distance(joints['right_wrist'][:2], joints['left_wrist'][:2]),
            'hand_sep_norm': calculate_distance(joints['right_wrist'][:2], joints['left_wrist'][:2]) / shoulder_width,
            'vertical_gap': joints['left_wrist'][1] - joints['right_wrist'][1],
            'vertical_gap_norm': (joints['left_wrist'][1] - joints['right_wrist'][1]) / shoulder_width,

            # THUMB FLICK FEATURES
            'left_thumb_x': joints['left_thumb'][0],
            'left_thumb_y': joints['left_thumb'][1],
            'pinch_dist': calculate_distance(joints['left_thumb'][:2], joints['left_index'][:2]),
            'hand_width': calculate_distance(joints['left_thumb'][:2], joints['left_pinky'][:2]),
            'thumb_extension': calculate_distance(joints['left_wrist'][:2], joints['left_thumb'][:2]),

            # KNEE FEATURES (front view)
            'right_knee_x': joints['right_knee'][0],
            'left_knee_x': joints['left_knee'][0],
            'right_ankle_x': joints['right_ankle'][0],
            'left_ankle_x': joints['left_ankle'][0],
            'knee_width': abs(joints['right_knee'][0] - joints['left_knee'][0]),
            'ankle_width': abs(joints['right_ankle'][0] - joints['left_ankle'][0]),
            'hip_width': abs(joints['right_hip'][0] - joints['left_hip'][0]),
            'valgus_ratio': abs(joints['right_knee'][0] - joints['left_knee'][0]) / max(abs(joints['right_ankle'][0] - joints['left_ankle'][0]), 1e-4),
            'right_knee_offset': 0.0,
            'left_knee_offset': 0.0,
            'knee_width_norm': abs(joints['right_knee'][0] - joints['left_knee'][0]) / shoulder_width,
            'ankle_width_norm': abs(joints['right_ankle'][0] - joints['left_ankle'][0]) / shoulder_width,
            'hip_width_norm': abs(joints['right_hip'][0] - joints['left_hip'][0]) / shoulder_width,
            'ankle_hip_ratio': abs(joints['right_ankle'][0] - joints['left_ankle'][0]) / max(abs(joints['right_hip'][0] - joints['left_hip'][0]), 1e-4),
            # Knee center position (for knee velocity / leg drive)
            'knee_mid_y': knee_mid[1],
            'knee_mid_x': knee_mid[0],
            'right_knee_y': joints['right_knee'][1],
            'left_knee_y': joints['left_knee'][1],
            
            # RELEASE HEIGHT
            'release_height': (joints['nose'][1] - joints['right_wrist'][1]) / torso_len,
            
            # SET POINT HEIGHT FEATURES
            'wrist_nose_diff': (joints['nose'][1] - joints['right_wrist'][1]) / torso_len,
            'wrist_shoulder_diff': (joints['right_shoulder'][1] - joints['right_wrist'][1]) / torso_len,
            'wrist_forehead_ratio': (joints['nose'][1] - joints['right_wrist'][1]) / torso_len,
            
            # FOREARM ANGLE
            'forearm_dx': joints['right_wrist'][0] - joints['right_elbow'][0],
            'forearm_dy': joints['right_wrist'][1] - joints['right_elbow'][1],
            'forearm_dz': joints['right_wrist'][2] - joints['right_elbow'][2],

            # Z-GAP (shoulder depth minus wrist depth; positive = wrist pushed forward)
            'shoulder_wrist_z_gap': joints['right_shoulder'][2] - joints['right_wrist'][2],

            # NORMALIZATION FACTORS
            'shoulder_width': shoulder_width,
            'torso_len': torso_len,
            'hip_center_y': (joints['right_hip'][1] + joints['left_hip'][1]) / 2,
            'nose_y': joints['nose'][1],
            'nose_x': joints['nose'][0],
        }

        # ============================================================
        # SIDE-VIEW FEATURES (new)
        # ============================================================

        # --- Torso lean (angle from vertical) ---
        torso_vec = shoulder_mid[:2] - hip_mid[:2]  # hip → shoulder in 2D
        # Angle from vertical: 0 = perfectly upright, positive = leaning
        # vertical in image coords is (0, -1) but we use abs(dx)/abs(dy)
        torso_lean = np.degrees(np.arctan2(
            np.abs(torso_vec[0]),
            np.abs(torso_vec[1]) + 1e-8
        ))
        frame_features['torso_lean_deg'] = float(torso_lean)

        # --- Head-torso offset (head drifting forward/back) ---
        frame_features['head_torso_offset_x'] = (joints['nose'][0] - shoulder_mid[0]) / shoulder_width

        # --- Wrist-to-torso perpendicular distance (ball path tightness) ---
        wrist_torso_dist = point_line_dist_2d(joints['right_wrist'], hip_mid, shoulder_mid)
        frame_features['wrist_torso_dist_norm'] = wrist_torso_dist / shoulder_width

        # --- Set point depth (wrist x relative to face/shoulder) ---
        # Positive = wrist IN FRONT of face (in x direction)
        # Sign convention: depends on which way shooter faces, but relative metrics
        # are consistent within a single video
        frame_features['wrist_face_depth_norm'] = (joints['right_wrist'][0] - joints['nose'][0]) / shoulder_width
        frame_features['wrist_shoulder_x_diff'] = (joints['right_wrist'][0] - joints['right_shoulder'][0]) / shoulder_width
        frame_features['elbow_wrist_x_offset'] = (joints['right_elbow'][0] - joints['right_wrist'][0]) / shoulder_width

        # --- Hip over ankle (stacking measure) ---
        frame_features['hip_over_ankle_x'] = (hip_mid[0] - ankle_mid[0]) / shoulder_width

        # --- Knee travel / shin angle (sagittal loading) ---
        # Right knee over right ankle/toe
        r_toe = joints['right_foot_index'][:2] if 'right_foot_index' in joints else joints['right_ankle'][:2]
        frame_features['knee_over_toe_x'] = (joints['right_knee'][0] - joints['right_ankle'][0]) / shoulder_width

        # Shin angle from vertical (right leg)
        shin_vec = joints['right_knee'][:2] - joints['right_ankle'][:2]
        shin_angle = np.degrees(np.arctan2(
            np.abs(shin_vec[0]),
            np.abs(shin_vec[1]) + 1e-8
        ))
        frame_features['shin_angle_deg'] = float(shin_angle)

        # Knee angle (for extension timing)
        frame_features['knee_angle_right'] = angle2d(
            joints['right_hip'][:2],
            joints['right_knee'][:2],
            joints['right_ankle'][:2]
        )

        # --- Twist proxy from side view (apparent width changes) ---
        # When body rotates, the apparent L-R distance shrinks/grows
        frame_features['shoulder_line_length_norm'] = shoulder_width / torso_len  # already have shoulder_width
        hip_line_length = abs(joints['right_hip'][0] - joints['left_hip'][0])
        frame_features['hip_line_length_norm'] = hip_line_length / torso_len
        frame_features['shoulder_hip_length_ratio'] = shoulder_width / max(hip_line_length, 1e-6)

        # --- Elbow-shoulder vertical deviation ---
        frame_features['elbow_shoulder_diff_y_norm'] = (joints['right_elbow'][1] - joints['right_shoulder'][1]) / torso_len

        # --- Landmark visibility (for reliability gating) ---
        frame_features['vis_left_shoulder'] = get_visibility(mp_lm.LEFT_SHOULDER)
        frame_features['vis_right_shoulder'] = get_visibility(mp_lm.RIGHT_SHOULDER)
        frame_features['vis_left_hip'] = get_visibility(mp_lm.LEFT_HIP)
        frame_features['vis_right_hip'] = get_visibility(mp_lm.RIGHT_HIP)

        # ============================================================
        # TWIST ANGLES (existing, using x/z)
        # ============================================================

        foot_vec = np.array([joints['right_ankle'][0] - joints['left_ankle'][0], 
                           joints['right_ankle'][2] - joints['left_ankle'][2]])
        foot_angle = np.arctan2(foot_vec[1], foot_vec[0])

        hip_vec = np.array([joints['right_hip'][0] - joints['left_hip'][0], 
                          joints['right_hip'][2] - joints['left_hip'][2]])
        hip_angle = np.arctan2(hip_vec[1], hip_vec[0])

        shoulder_vec_xz = np.array([joints['right_shoulder'][0] - joints['left_shoulder'][0], 
                                    joints['right_shoulder'][2] - joints['left_shoulder'][2]])
        shoulder_angle = np.arctan2(shoulder_vec_xz[1], shoulder_vec_xz[0])

        frame_features['stance_angle'] = np.degrees(foot_angle)
        frame_features['hip_absolute_yaw'] = np.degrees(hip_angle)
        frame_features['shoulder_absolute_yaw'] = np.degrees(shoulder_angle)
        frame_features['hip_twist_from_feet'] = normalize_angle(np.degrees(hip_angle - foot_angle))
        frame_features['shoulder_twist_from_feet'] = normalize_angle(np.degrees(shoulder_angle - foot_angle))
        frame_features['twist_yaw_deg'] = normalize_angle(np.degrees(shoulder_angle - hip_angle))

        # Knee offsets (existing)
        def get_deviation(hip, knee, ankle):
            if abs(ankle[1] - hip[1]) < 1e-4:
                return 0.0
            progress = (knee[1] - hip[1]) / (ankle[1] - hip[1])
            expected_x = hip[0] + progress * (ankle[0] - hip[0])
            return knee[0] - expected_x

        r_dev = get_deviation(joints['right_hip'], joints['right_knee'], joints['right_ankle'])
        l_dev = get_deviation(joints['left_hip'], joints['left_knee'], joints['left_ankle'])
        frame_features['right_knee_offset'] = r_dev
        frame_features['left_knee_offset'] = -l_dev

        frame_features_list.append(frame_features)

    pose.close()

    if len(frame_features_list) == 0:
        print("   ❌ No keypoints extracted")
        return None

    # Create dataframe
    df = pd.DataFrame(frame_features_list)
    
    # ============================================================
    # SMOOTHED COLUMNS FOR PHASE DETECTION (existing)
    # ============================================================
    
    df['r_wrist_y_s'] = smooth_signal(df['r_wrist_y_px'].to_numpy(), win=9, poly=2)
    df['r_wrist_x_s'] = smooth_signal(df['r_wrist_x_px'].to_numpy(), win=9, poly=2)
    df['r_elbow_s'] = smooth_signal(df['r_elbow_angle'].to_numpy(), win=9, poly=2)
    df['r_elbow_y_s'] = smooth_signal(df['r_elbow_y_px'].to_numpy(), win=9, poly=2)
    df['r_shoulder_y_s'] = smooth_signal(df['r_shoulder_y_px'].to_numpy(), win=9, poly=2)
    
    df['wrist_y_s'] = df['r_wrist_y_s']
    df['wrist_x_s'] = df['r_wrist_x_s']
    df['elbow_s'] = df['r_elbow_s']
    df['elbow_y_s'] = df['r_elbow_y_s']
    df['shoulder_y_s'] = df['r_shoulder_y_s']
    
    # ============================================================
    # VELOCITY COLUMNS (FIX: both from smoothed signals now)
    # ============================================================
    
    df['wrist_vy'] = np.gradient(df['wrist_y_s'].to_numpy())
    df['wrist_vx'] = np.gradient(df['wrist_x_s'].to_numpy())  # FIXED: was diff(raw), now gradient(smoothed)
    df['wrist_ay'] = np.gradient(df['wrist_vy'].to_numpy())
    df['elbow_v'] = np.gradient(df['elbow_s'].to_numpy())
    
    # Keep old column names for backward compatibility
    df['wrist_y_px_smooth'] = df['wrist_y_s']
    df['elbow_px_smooth'] = df['elbow_s']
    
    # Smooth signals for feature analysis
    df['right_wrist_y_smooth'] = df['right_wrist_y'].rolling(window=5, center=True, min_periods=1).median()
    df['elbow_angle_smooth'] = df['elbow_angle_3d'].rolling(window=5, center=True, min_periods=1).mean()
    df['pinch_dist_smooth'] = df['pinch_dist'].rolling(window=3, center=True, min_periods=1).mean()
    df['hand_sep_norm_smooth'] = df['hand_sep_norm'].rolling(window=5, center=True, min_periods=1).median()
    df['vertical_gap_norm_smooth'] = df['vertical_gap_norm'].rolling(window=5, center=True, min_periods=1).median()
    df['valgus_ratio_smooth'] = df['valgus_ratio'].rolling(window=3, center=True, min_periods=1).mean()

    # Additional velocity columns
    df['wrist_jerk'] = df['wrist_ay'].diff()
    df['elbow_rate'] = df['elbow_angle_smooth'].diff()
    df['thumb_flick_vel'] = df['pinch_dist_smooth'].diff() * 100

    # ============================================================
    # BODY SEGMENT VELOCITIES (for kinetic chain / fluidity)
    # ============================================================

    # Hip velocity — core kinetic chain signal
    df['hip_y_s'] = smooth_signal(df['hip_mid_y'].to_numpy(), win=9, poly=2)
    df['hip_x_s'] = smooth_signal(df['hip_mid_x'].to_numpy(), win=9, poly=2)
    df['hip_vy'] = np.gradient(df['hip_y_s'].to_numpy())   # negative = moving up
    df['hip_vx'] = np.gradient(df['hip_x_s'].to_numpy())
    df['hip_ay'] = np.gradient(df['hip_vy'].to_numpy())

    # Shoulder position velocity (mid-body link in chain)
    df['shoulder_pos_y_s'] = smooth_signal(df['shoulder_mid_y'].to_numpy(), win=9, poly=2)
    df['shoulder_pos_vy'] = np.gradient(df['shoulder_pos_y_s'].to_numpy())

    # Knee position velocity (leg drive signal — distinct from knee_vy which is angle)
    df['knee_pos_y_s'] = smooth_signal(df['knee_mid_y'].to_numpy(), win=9, poly=2)
    df['knee_pos_vy'] = np.gradient(df['knee_pos_y_s'].to_numpy())
    df['knee_pos_ay'] = np.gradient(df['knee_pos_vy'].to_numpy())

    # Elbow position velocity (arm flow, separate from elbow angle rate)
    df['elbow_pos_y_s'] = smooth_signal(df['r_elbow_y_px'].to_numpy(), win=9, poly=2)
    df['elbow_pos_vy'] = np.gradient(df['elbow_pos_y_s'].to_numpy())
    
    # FOREARM ANGLE FROM VERTICAL
    df['forearm_angle_from_vertical'] = np.degrees(np.arctan2(
        np.abs(df['forearm_dx']),
        np.abs(df['forearm_dy'])
    ))
    
    # WRIST FORWARD-TO-UP RATIO (now using consistent smoothed velocities)
    epsilon = 0.001
    df['wrist_forward_up_ratio'] = np.abs(df['wrist_vx']) / (np.abs(df['wrist_vy']) + epsilon)
    df['wrist_forward_up_ratio_smooth'] = df['wrist_forward_up_ratio'].rolling(window=3, center=True, min_periods=1).median()
    
    # RELEASE ANGLE (from smoothed velocity vector)
    # In image coords: vy negative = rising, vx positive = forward
    # angle = atan2(|vy|, |vx|) gives angle above horizontal (90 = straight up, 0 = flat)
    df['release_angle_deg'] = np.degrees(np.arctan2(
        np.abs(df['wrist_vy']),
        np.abs(df['wrist_vx']) + 1e-6
    ))
    
    # Delta columns
    df['delta_right_wrist_y'] = df['right_wrist_y'].diff()
    df['delta_left_wrist_y'] = df['left_wrist_y'].diff()

    # ============================================================
    # SIDE-VIEW DERIVED COLUMNS
    # ============================================================
    
    # Wrist-torso distance slope (rate of change)
    df['wrist_torso_dist_norm_smooth'] = df['wrist_torso_dist_norm'].rolling(window=5, center=True, min_periods=1).median()
    df['wrist_torso_slope'] = df['wrist_torso_dist_norm_smooth'].diff()
    
    # Torso lean smoothed
    df['torso_lean_deg_smooth'] = df['torso_lean_deg'].rolling(window=5, center=True, min_periods=1).median()
    
    # Guide hand separation velocity
    df['sep_velocity'] = df['hand_sep_norm_smooth'].diff()
    
    # Knee velocity (for fluidity / coupling analysis)
    knee_y_smooth = smooth_signal(df['knee_angle_right'].to_numpy(), win=7, poly=2) if 'knee_angle_right' in df.columns else np.zeros(len(df))
    df['knee_angle_smooth'] = knee_y_smooth
    df['knee_vy'] = np.gradient(knee_y_smooth)  # rate of knee angle change
    
    # Shin angle smoothed
    df['shin_angle_deg_smooth'] = df['shin_angle_deg'].rolling(window=5, center=True, min_periods=1).median()
    
    # Shoulder/hip line length smoothed (for twist proxy)
    df['shoulder_line_length_norm_smooth'] = df['shoulder_line_length_norm'].rolling(window=5, center=True, min_periods=1).median()
    df['hip_line_length_norm_smooth'] = df['hip_line_length_norm'].rolling(window=5, center=True, min_periods=1).median()

    # ============================================================
    # RELEASE FRAME DETECTION (existing)
    # ============================================================
    
    # ============================================================
    # BALL DETECTION — merge ball position per frame
    # ============================================================
    ball_detections = get_ball_detections(video_path)
    if ball_detections:
        ball_cx = df['frame'].map(lambda f: ball_detections[f][0] if f in ball_detections else np.nan)
        ball_cy = df['frame'].map(lambda f: ball_detections[f][1] if f in ball_detections else np.nan)
        df['ball_x'] = ball_cx
        df['ball_y'] = ball_cy
        df['ball_wrist_dist'] = np.sqrt(
            (df['ball_x'] - df['r_wrist_x_px']) ** 2 +
            (df['ball_y'] - df['r_wrist_y_px']) ** 2
        )
        df['ball_wrist_dist_norm'] = df['ball_wrist_dist'] / df['shoulder_width'].replace(0, np.nan)
    else:
        df['ball_x'] = np.nan
        df['ball_y'] = np.nan
        df['ball_wrist_dist'] = np.nan
        df['ball_wrist_dist_norm'] = np.nan

    release_idx = detect_release_frame(df, ball_detections=ball_detections if ball_detections else None, use_classifier=False)
    if release_idx is not None:
        release_frame = int(df.iloc[release_idx]['frame'])
        print(f"   🎯 Release detected at frame {release_frame}")
        df['release_frame_detected'] = release_frame
        df['is_release_frame'] = (df['frame'] == release_frame).astype(int)
    else:
        print("   ⚠️ Could not detect release frame")
        df['release_frame_detected'] = -1
        df['is_release_frame'] = 0

    print(f"   ✅ Extracted {len(df)} frames with {len(df.columns)} features (front + side view)")
    return df