"""
Pose detection utilities for image normalization.

Adapted from /Users/yonatan/Desktop/dminti/server-4/server/image_processing/pose_crop.py
Uses MediaPipe PoseLandmarker (Tasks API) to detect the most prominent person
and estimate head-top (scalp) position and horizontal center.
"""
import os
import threading
import numpy as np
from PIL import Image
import mediapipe as mp

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Minimum visibility threshold for a landmark to be considered reliable
MIN_VISIBILITY = 0.3

# Path to the bundled model file (relative to this module)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

# Module-level singleton and lock for thread safety
_pose_landmarker = None
_pose_lock = threading.Lock()


def _get_pose_landmarker():
    """Get or initialize the MediaPipe PoseLandmarker singleton."""
    global _pose_landmarker
    if _pose_landmarker is None:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=5,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        _pose_landmarker = PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def _pick_best_person(result, img_w, img_h):
    """
    From multiple detected poses, pick the most prominent person.
    Scoring: visibility * 0.3 + size * 0.5 + centrality * 0.2
    """
    if not result.pose_landmarks:
        return None

    best_score = -1
    best_landmarks = None

    for pose_landmarks in result.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.visibility) for lm in pose_landmarks]

        key_indices = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER]
        vis_score = sum(landmarks[i][2] for i in key_indices) / len(key_indices)

        shoulder_width = abs(landmarks[RIGHT_SHOULDER][0] - landmarks[LEFT_SHOULDER][0])
        size_score = shoulder_width

        nose_x, nose_y = landmarks[NOSE][0], landmarks[NOSE][1]
        dist_from_center = ((nose_x - 0.5) ** 2 + (nose_y - 0.5) ** 2) ** 0.5
        centrality_score = max(0, 1 - dist_from_center)

        score = vis_score * 0.3 + size_score * 0.5 + centrality_score * 0.2

        if score > best_score:
            best_score = score
            best_landmarks = landmarks

    return best_landmarks


def detect_pose(image_pil):
    """
    Run MediaPipe PoseLandmarker on a PIL Image (thread-safe).

    Returns:
        List of (x_norm, y_norm, visibility) tuples for 33 landmarks
        of the most prominent person, or None if no person detected.
    """
    image_rgb = np.array(image_pil.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with _pose_lock:
        landmarker = _get_pose_landmarker()
        result = landmarker.detect(mp_image)

    return _pick_best_person(result, image_pil.size[0], image_pil.size[1])


def estimate_head_top_y(landmarks, img_h, crown_margin=2.6):
    """
    Estimate the Y-coordinate (in pixels) of the top of the scalp.

    Uses eye positions and nose-to-eye distance with the crown_margin
    multiplier from the existing pose_crop.py logic.

    Returns:
        (head_top_y_px, method) or (None, None) if landmarks insufficient.
    """
    nose_y = landmarks[NOSE][1] * img_h
    nose_vis = landmarks[NOSE][2]
    left_eye_y = landmarks[LEFT_EYE][1] * img_h
    left_eye_vis = landmarks[LEFT_EYE][2]
    right_eye_y = landmarks[RIGHT_EYE][1] * img_h
    right_eye_vis = landmarks[RIGHT_EYE][2]
    left_shoulder_y = landmarks[LEFT_SHOULDER][1] * img_h
    left_shoulder_vis = landmarks[LEFT_SHOULDER][2]
    right_shoulder_y = landmarks[RIGHT_SHOULDER][1] * img_h
    right_shoulder_vis = landmarks[RIGHT_SHOULDER][2]

    if nose_vis < MIN_VISIBILITY:
        return None, None

    # Eye position: use highest visible eye
    eye_y = None
    if left_eye_vis >= MIN_VISIBILITY and right_eye_vis >= MIN_VISIBILITY:
        eye_y = min(left_eye_y, right_eye_y)
    elif left_eye_vis >= MIN_VISIBILITY:
        eye_y = left_eye_y
    elif right_eye_vis >= MIN_VISIBILITY:
        eye_y = right_eye_y

    if eye_y is None:
        # No eyes visible — try estimating from nose + shoulders
        if left_shoulder_vis >= MIN_VISIBILITY or right_shoulder_vis >= MIN_VISIBILITY:
            shoulder_y_avg = 0
            count = 0
            if left_shoulder_vis >= MIN_VISIBILITY:
                shoulder_y_avg += left_shoulder_y
                count += 1
            if right_shoulder_vis >= MIN_VISIBILITY:
                shoulder_y_avg += right_shoulder_y
                count += 1
            shoulder_y_avg /= count
            face_height = abs(nose_y - shoulder_y_avg) * 0.3
            head_top_y = nose_y - face_height * (crown_margin + 1)
            return max(0, head_top_y), "nose_shoulder_fallback"
        return None, None

    face_height = abs(nose_y - eye_y)

    # Fallback if eyes are too close to nose
    if face_height < 5:
        shoulder_y_avg = 0
        count = 0
        if left_shoulder_vis >= MIN_VISIBILITY:
            shoulder_y_avg += left_shoulder_y
            count += 1
        if right_shoulder_vis >= MIN_VISIBILITY:
            shoulder_y_avg += right_shoulder_y
            count += 1
        if count > 0:
            shoulder_y_avg /= count
            face_height = abs(nose_y - shoulder_y_avg) * 0.3

    head_top_y = eye_y - face_height * crown_margin
    return max(0, head_top_y), "pose"


def estimate_person_center_x(landmarks, img_w):
    """
    Return horizontal center of person in pixel coordinates.
    Prefers shoulder midpoint, falls back to nose.
    """
    ls_x, _, ls_vis = landmarks[LEFT_SHOULDER]
    rs_x, _, rs_vis = landmarks[RIGHT_SHOULDER]

    if ls_vis >= MIN_VISIBILITY and rs_vis >= MIN_VISIBILITY:
        return (ls_x + rs_x) / 2 * img_w

    return landmarks[NOSE][0] * img_w
