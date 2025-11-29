import argparse
import asyncio
import json
import signal
import threading
import time
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import websockets
import json
import argparse
import threading
import logging
import os
from pathlib import Path
from contextlib import suppress
from typing import Optional, Dict, Any

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BROADCAST_INTERVAL = 1 / 45  # seconds (faster updates reduce perceived latency)
WEBSOCKET_PORT = 6789
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

running = True
payload_lock = threading.Lock()
latest_payload = {
    "ratio": 0.0,
    "open": 0.0,
    "pucker": 0.0,
    "wide": 0.0,
    "smile": 0.0,
    "tongue": 0.0,
    "shift": 0.0,
    "detected": False,
    "confidence": 0.0,
    "source": "none",
    "horizontal": 0.0,
    "emotion": "neutral",
    "emotion_score": 0.0,
}
clients: set[websockets.WebSocketServerProtocol] = set()
clients_lock = threading.Lock()
EMOTION_MODEL_WARNING_EMITTED = False

# --- Constants and Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BROADCAST_INTERVAL = 1 / 45  # ~45 FPS
WEBSOCKET_PORT = 6789

def load_mouth_cascade() -> cv2.CascadeClassifier:
    local_cascade = Path(__file__).with_name("haarcascade_mouth.xml")
    if local_cascade.exists():
        cascade = cv2.CascadeClassifier(str(local_cascade))
        if not cascade.empty():
            return cascade
    fallback_path = Path(cv2.data.haarcascades) / "haarcascade_mcs_mouth.xml"
    cascade = cv2.CascadeClassifier(str(fallback_path))
    if cascade.empty():
        raise RuntimeError("Unable to load a mouth cascade classifier. Please verify the XML path.")
    return cascade
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


mouth_cascade = load_mouth_cascade()
# --- Model Loading ---
face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
if face_cascade.empty():
    raise RuntimeError("Unable to load face cascade classifier. Check your OpenCV installation.")

try:
    import mediapipe as mp
    print("[SETUP] MediaPipe found.")
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[SETUP] MediaPipe not found. Falling back to Haar cascades.")
    MEDIAPIPE_AVAILABLE = False

def load_emotion_model() -> Optional[object]:
    """Load the optional emotion recognition model if TensorFlow is available."""
    global EMOTION_MODEL_WARNING_EMITTED

    model_dir = Path(__file__).resolve().parent.parent / "Face-detection-model"
    json_path = model_dir / "emotiondetector.json"
    weights_path = model_dir / "emotiondetector.h5"

    if not json_path.exists() or not weights_path.exists():
        if not EMOTION_MODEL_WARNING_EMITTED:
            print(
                f"[EMOTION] Model files missing. Expected {json_path.name} and {weights_path.name} inside {model_dir}.",
                file=sys.stderr,
            )
            EMOTION_MODEL_WARNING_EMITTED = True
        print(f"[EMOTION] Model files missing. Emotion detection disabled.", file=sys.stderr)
        return None

    try:
        from tensorflow.keras.models import model_from_json  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if not EMOTION_MODEL_WARNING_EMITTED:
            print(f"[EMOTION] TensorFlow/Keras not available: {exc}. Emotion cues disabled.", file=sys.stderr)
            EMOTION_MODEL_WARNING_EMITTED = True
        print(f"[EMOTION] TensorFlow/Keras not available: {exc}. Emotion detection disabled.", file=sys.stderr)
        return None

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            model_json = handle.read()
        model = model_from_json(model_json)
        model.load_weights(str(weights_path))
        print("[EMOTION] Loaded emotion recognition model.")
        return model
    except Exception as exc:  # pragma: no cover - optional dependency
        if not EMOTION_MODEL_WARNING_EMITTED:
            print(f"[EMOTION] Failed to load emotion model: {exc}", file=sys.stderr)
            EMOTION_MODEL_WARNING_EMITTED = True
        print(f"[EMOTION] Failed to load emotion model: {exc}", file=sys.stderr)
        return None

smile_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(str(smile_cascade_path)) if smile_cascade_path.exists() else None
if smile_cascade is not None and smile_cascade.empty():
    smile_cascade = None

cap_lock = threading.Lock()
cap: Optional[cv2.VideoCapture] = None


emotion_model_lock = threading.Lock()
emotion_model = None
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
EMOTION_PREDICTION_INTERVAL = 0.9  # seconds between inferences
EMOTION_SMOOTHING = 0.65
EMOTION_DECAY = 0.92
EMOTION_IDLE_THRESHOLD = 0.12
EMOTION_MODEL_WARNING_EMITTED = False


NOSE_TIP_INDEX = 1
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14
LEFT_MOUTH_INDEX = 61
RIGHT_MOUTH_INDEX = 291
OUTER_MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]
INNER_MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]


def _landmark_to_xy(landmarks, index: int, width: int, height: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def compute_mediapipe_mouth_metrics(
    landmarks, width: int, height: int) -> Dict[str, Any]:
    """Return core mouth ratios plus outer/inner landmark rings for downstream cues."""

    left = _landmark_to_xy(landmarks, LEFT_MOUTH_INDEX, width, height)
    right = _landmark_to_xy(landmarks, RIGHT_MOUTH_INDEX, width, height)
    upper = _landmark_to_xy(landmarks, UPPER_LIP_INDEX, width, height)
    lower = _landmark_to_xy(landmarks, LOWER_LIP_INDEX, width, height)

    horizontal = np.linalg.norm(left - right)
    if horizontal <= 1e-6:
        return {}
    vertical = np.linalg.norm(upper - lower)
    ratio = float(vertical / horizontal)

    norm_factor = float(max(width, height)) or 1.0
    horizontal_ratio = float(np.clip(horizontal / norm_factor, 0.0, 1.0))

    outer_ring = [_landmark_to_xy(landmarks, idx, width, height) for idx in OUTER_MOUTH_INDICES]
    inner_ring = [_landmark_to_xy(landmarks, idx, width, height) for idx in INNER_MOUTH_INDICES]

    xs = [pt[0] for pt in outer_ring]
    ys = [pt[1] for pt in outer_ring]
    x0 = int(max(0, min(xs)))
    y0 = int(max(0, min(ys)))
    x1 = int(min(width - 1, max(xs)))
    y1 = int(min(height - 1, max(ys)))
    bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    center_y = float((upper[1] + lower[1]) * 0.5)
    corner_y = float((left[1] + right[1]) * 0.5)
    smile_delta = float(np.clip((center_y - corner_y) / float(max(height, 1)), -0.2, 0.2))

    outer_int = [(int(round(pt[0])), int(round(pt[1]))) for pt in outer_ring]
    inner_int = [(int(round(pt[0])), int(round(pt[1]))) for pt in inner_ring]
    return {
        "ratio": ratio,
        "bbox": bbox,
        "horizontal": horizontal_ratio,
        "smile_delta": smile_delta,
        "outer_ring": outer_int,
        "inner_ring": inner_int,
    }


def normalize_ratio(raw_ratio: float, baseline: float, spread: float) -> float:
    return float(np.clip((raw_ratio - baseline) / spread, 0.0, 1.0))


def initialise_camera(index: int) -> bool:
    """Initialise the OpenCV capture device with low latency settings."""

    global cap
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    new_cap = cv2.VideoCapture(index, backend)
    if not new_cap.isOpened():
        new_cap.release()
        return False

    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    with suppress(cv2.error):
        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    with suppress(cv2.error):
        new_cap.set(cv2.CAP_PROP_FPS, 45)

    with cap_lock:
        if cap is not None:
            cap.release()
        cap = new_cap
    return True


def estimate_tongue_presence(
    frame: np.ndarray,
    polygon: Optional[list[tuple[int, int]]] = None,
    rect: Optional[tuple[int, int, int, int]] = None,
) -> float:
    if polygon is None and rect is None:
        return 0.0

    height, width = frame.shape[:2]
    if polygon is not None and len(polygon) > 0:
        poly = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        if w <= 0 or h <= 0:
            return 0.0
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        w = int(np.clip(w, 1, width - x))
        h = int(np.clip(h, 1, height - y))
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = poly - np.array([x, y])
        cv2.fillPoly(mask, [shifted], 255)
    elif rect is not None:
        x, y, w, h = rect
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        w = int(np.clip(w, 1, width - x))
        h = int(np.clip(h, 1, height - y))
        if w <= 0 or h <= 0:
            return 0.0
        mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        return 0.0

    region = frame[y : y + h, x : x + w]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    active = mask > 0
    if not np.any(active):
        return 0.0
    sample = hsv[active]
    if sample.size == 0:
        return 0.0

    hue = sample[:, 0].astype(np.float32) / 180.0
    sat = sample[:, 1].astype(np.float32) / 255.0
    val = sample[:, 2].astype(np.float32) / 255.0

    red_band = ((hue < 0.075) | (hue > 0.93)) & (sat > 0.28) & (val > 0.22)
    pink_band = (hue >= 0.075) & (hue < 0.16) & (sat > 0.24) & (val > 0.32)
    ratio = float(np.count_nonzero(red_band | pink_band) / max(1, sample.shape[0]))
    return float(np.clip((ratio - 0.08) / 0.24, 0.0, 1.0))


def estimate_smile_with_cascade(gray_frame: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    if smile_cascade is None:
        return 0.0
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return 0.0
    roi = gray_frame[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    smiles = smile_cascade.detectMultiScale(
        roi,
        scaleFactor=1.22,
        minNeighbors=12,
        minSize=(max(18, w // 5), max(12, h // 5)),
    )
    if len(smiles) == 0:
        return 0.0
    best = max(smiles, key=lambda r: r[2] * r[3])
    coverage = float((best[2] * best[3]) / float(max(w * h, 1)))
    return float(np.clip((coverage - 0.04) / 0.18, 0.0, 1.0))


def detection_loop(show_preview: bool, use_mediapipe: bool) -> None:
    global latest_payload, running, EMOTION_MODEL_WARNING_EMITTED, emotion_model
    global latest_payload, running, emotion_model
    print("[DETECTION] Detection loop started.")

    face_mesh = None
    if use_mediapipe:
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            print("[DETECTION] MediaPipe FaceMesh initialised.")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[DETECTION] Failed to initialise MediaPipe FaceMesh: {exc}. Falling back to cascades.")
            face_mesh = None

    smoothing_open = 0.18
    smoothing_pucker = 0.22
    smoothing_wide = 0.24
    smoothing_smile = 0.28
    smoothing_tongue = 0.35
    smoothing_shift = 0.35
    previous_open = 0.0
    previous_pucker = 0.0
    previous_wide = 0.0
    previous_smile = 0.0
    previous_tongue = 0.0
    previous_shift = 0.0

    mp_open_baseline = 0.06
    mp_open_range = 0.18
    mp_width_baseline: Optional[float] = None
    mp_width_range = 0.14
    mp_smile_baseline: Optional[float] = None
    mp_smile_range = 0.05
    mp_center_baseline: Optional[float] = None
    mp_shift_range = 0.26

    cascade_open_baseline = 0.06
    cascade_open_range = 0.18
    cascade_width_baseline: Optional[float] = None
    cascade_width_range = 0.14
    cascade_center_baseline: Optional[float] = None
    cascade_shift_range = 0.32

    no_detection_frames = 0
    last_face: Optional[tuple[int, int, int, int]] = None
    mouth_landmarks: Optional[list[tuple[int, int]]] = None
    inner_mouth_outline: Optional[list[tuple[int, int]]] = None
    emotion_label_state = "neutral"
    emotion_score_state = 0.0
    emotion_last_inference = 0.0
    emotion_last_face: Optional[tuple[int, int, int, int]] = None
    local_emotion_model = None
    with emotion_model_lock:
        local_emotion_model = emotion_model

    try:
        while running:
            with cap_lock:
                local_cap = cap
            if local_cap is None:
                time.sleep(0.05)
                continue

            ret, frame = local_cap.read()
            if not ret:
                no_detection_frames += 1
                if no_detection_frames > 10:
                    with payload_lock:
                        latest_payload = {
                            "ratio": 0.0,
                            "open": 0.0,
                            "pucker": 0.0,
                            "wide": 0.0,
                            "smile": 0.0,
                            "tongue": 0.0,
                            "shift": 0.0,
                            "detected": False,
                            "confidence": 0.0,
                            "source": "none",
                            "horizontal": 0.0,
                            "emotion": "neutral",
                            "emotion_score": 0.0,
                        }
                    time.sleep(0.05)
                else:
                    time.sleep(0.01)
                continue

            height, width = frame.shape[:2]
            detection_source = "none"
            mouth_landmarks = None
            inner_mouth_outline = None
            horizontal_ratio: Optional[float] = None
            best_mouth_rect: Optional[tuple[int, int, int, int]] = None

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized_gray = cv2.equalizeHist(gray_frame)
            now = time.time()
            if local_emotion_model is None:
                emotion_label_state = "neutral"
                emotion_score_state = 0.0
            elif now - emotion_last_inference > EMOTION_PREDICTION_INTERVAL * 2.2:
                emotion_score_state *= EMOTION_DECAY
                if emotion_score_state < EMOTION_IDLE_THRESHOLD:
                    emotion_label_state = "neutral"
                    emotion_score_state = 0.0

            open_norm = previous_open * 0.45
            pucker_norm = previous_pucker * 0.45
            wide_norm = previous_wide * 0.45
            smile_norm = previous_smile * 0.5
            tongue_norm = previous_tongue * 0.5
            shift_norm = previous_shift * 0.55

            if face_mesh is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if MEDIAPIPE_AVAILABLE and results and results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    metrics = compute_mediapipe_mouth_metrics(landmarks, width, height)

                    if metrics:
                        open_ratio, bbox, horizontal, smile_delta, outer_points, inner_points = (
                            metrics.get("ratio"), metrics.get("bbox"), metrics.get("horizontal"), metrics.get("smile_delta"), metrics.get("outer_ring"), metrics.get("inner_ring")
                        )
                        detection_source = "mediapipe"
                        best_mouth_rect = bbox
                        horizontal_ratio = horizontal
                        mouth_landmarks = outer_points or None
                        inner_mouth_outline = inner_points or None
                        no_detection_frames = 0

                        if open_ratio is not None:
                            if open_ratio < mp_open_baseline + mp_open_range * 0.5:
                                mp_open_baseline = float(np.clip(mp_open_baseline * 0.9 + open_ratio * 0.1, 0.02, 0.25))
                            raw_open = normalize_ratio(open_ratio, mp_open_baseline, mp_open_range)
                            open_norm = smoothing_open * previous_open + (1 - smoothing_open) * raw_open

                        if horizontal is not None:
                            if mp_width_baseline is None:
                                mp_width_baseline = horizontal
                            else:
                                mp_width_baseline = mp_width_baseline * 0.92 + horizontal * 0.08

                            raw_pucker = 0.0
                            raw_wide = 0.0
                            if mp_width_baseline is not None:
                                raw_pucker = float(np.clip((mp_width_baseline - horizontal) / mp_width_range, 0.0, 1.0))
                                raw_wide = float(np.clip((horizontal - mp_width_baseline) / mp_width_range, 0.0, 1.0))
                            pucker_norm = smoothing_pucker * previous_pucker + (1 - smoothing_pucker) * raw_pucker
                            wide_norm = smoothing_wide * previous_wide + (1 - smoothing_wide) * raw_wide

                            if smile_delta is not None:
                                if mp_smile_baseline is None:
                                    mp_smile_baseline = smile_delta
                                else:
                                    if smile_delta <= (mp_smile_baseline + 0.01):
                                        mp_smile_baseline = mp_smile_baseline * 0.9 + smile_delta * 0.1
                                    else:
                                        mp_smile_baseline = mp_smile_baseline * 0.98 + smile_delta * 0.02
                                smile_raw = float(np.clip((smile_delta - (mp_smile_baseline or 0.0)) / mp_smile_range, 0.0, 1.0))
                                smile_norm = smoothing_smile * previous_smile + (1 - smoothing_smile) * smile_raw

                            left_corner = _landmark_to_xy(landmarks, LEFT_MOUTH_INDEX, width, height)
                            right_corner = _landmark_to_xy(landmarks, RIGHT_MOUTH_INDEX, width, height)
                            mouth_span = float(np.linalg.norm(left_corner - right_corner))
                            if mouth_span > 1e-3:
                                mouth_center = (left_corner + right_corner) * 0.5
                                nose_tip = _landmark_to_xy(landmarks, NOSE_TIP_INDEX, width, height)
                                center_offset = float((mouth_center[0] - nose_tip[0]) / mouth_span)
                                if mp_center_baseline is None:
                                    mp_center_baseline = center_offset
                                else:
                                    if abs(center_offset - mp_center_baseline) < 0.18:
                                        mp_center_baseline = mp_center_baseline * 0.92 + center_offset * 0.08
                                    else:
                                        mp_center_baseline = mp_center_baseline * 0.985 + center_offset * 0.015
                                shift_raw = float(np.clip((center_offset - (mp_center_baseline or 0.0)) / mp_shift_range, -1.0, 1.0))
                                shift_norm = smoothing_shift * previous_shift + (1 - smoothing_shift) * shift_raw

                        if inner_points:
                            tongue_raw = 0.0
                            if open_norm > 0.16:
                                tongue_raw = estimate_tongue_presence(frame, polygon=inner_points)
                                tongue_raw = estimate_tongue_presence(frame, polygon=outer_points)
                            tongue_norm = smoothing_tongue * previous_tongue + (1 - smoothing_tongue) * tongue_raw

            if detection_source == "none":
                gray = equalized_gray

                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(120, 120))
                if len(faces) == 0 and last_face is not None:
                    faces = np.array([last_face])

                for (fx, fy, fw, fh) in faces:
                    mouth_region_y = int(fy + fh * 0.45)
                    roi = gray[mouth_region_y : fy + fh, fx : fx + fw]
                    if roi.size == 0:
                        continue
                    roi_mouths = mouth_cascade.detectMultiScale(
                        roi,
                        scaleFactor=1.18,
                        minNeighbors=3,
                        minSize=(int(fw * 0.16), int(fh * 0.1)),
                    )
                    if len(roi_mouths) == 0:
                        continue
                    rx, ry, rw, rh = max(roi_mouths, key=lambda rect: rect[2] * rect[3])
                    candidate = (fx + rx, mouth_region_y + ry, rw, rh)
                    if best_mouth_rect is None or rw * rh > best_mouth_rect[2] * best_mouth_rect[3]:
                        best_mouth_rect = candidate
                        last_face = (fx, fy, fw, fh)
                    # Simple heuristic for mouth ROI based on face
                    best_mouth_rect = (fx, fy + int(fh * 0.6), fw, int(fh * 0.4))
                    last_face = (fx, fy, fw, fh)

                if best_mouth_rect is None:
                    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.18, minNeighbors=5, minSize=(50, 32))
                    if len(mouths) > 0:
                        best_mouth_rect = max(mouths, key=lambda rect: rect[2] * rect[3])
                        last_face = None

                if best_mouth_rect is not None:
                    x, y, w, h = best_mouth_rect
                    raw_ratio = h / float(max(w, 1))
                    if raw_ratio < cascade_open_baseline + cascade_open_range * 0.5:
                        cascade_open_baseline = float(np.clip(cascade_open_baseline * 0.9 + raw_ratio * 0.1, 0.02, 0.25))
                    raw_open = normalize_ratio(raw_ratio, cascade_open_baseline, cascade_open_range)
                    open_norm = smoothing_open * previous_open + (1 - smoothing_open) * raw_open

                    horizontal = float(np.clip(w / float(max(width, 1)), 0.0, 1.0))
                    horizontal_ratio = horizontal
                    if cascade_width_baseline is None:
                        cascade_width_baseline = horizontal
                    else:
                        cascade_width_baseline = cascade_width_baseline * 0.92 + horizontal * 0.08
                    raw_pucker = float(np.clip((cascade_width_baseline - horizontal) / cascade_width_range, 0.0, 1.0))
                    raw_wide = float(np.clip((horizontal - cascade_width_baseline) / cascade_width_range, 0.0, 1.0))
                    pucker_norm = smoothing_pucker * previous_pucker + (1 - smoothing_pucker) * raw_pucker
                    wide_norm = smoothing_wide * previous_wide + (1 - smoothing_wide) * raw_wide

                    smile_fallback = estimate_smile_with_cascade(gray, best_mouth_rect)
                    smile_source = max(raw_wide * 0.6, smile_fallback)
                    smile_norm = smoothing_smile * previous_smile + (1 - smoothing_smile) * smile_source

                    tongue_raw = 0.0
                    if open_norm > 0.2:
                        tongue_raw = estimate_tongue_presence(frame, rect=best_mouth_rect)
                    tongue_norm = smoothing_tongue * previous_tongue + (1 - smoothing_tongue) * tongue_raw

                    mouth_landmarks = [
                        (int(x), int(y)),
                        (int(x + w), int(y)),
                        (int(x + w), int(y + h)),
                        (int(x), int(y + h)),
                    ]
                    inner_mouth_outline = mouth_landmarks
                    mouth_center_x = float((x + (w * 0.5)) / max(width, 1))
                    center_offset = mouth_center_x - 0.5
                    if cascade_center_baseline is None:
                        cascade_center_baseline = center_offset
                    else:
                        if abs(center_offset - cascade_center_baseline) < 0.22:
                            cascade_center_baseline = cascade_center_baseline * 0.9 + center_offset * 0.1
                        else:
                            cascade_center_baseline = cascade_center_baseline * 0.97 + center_offset * 0.03
                    shift_raw = float(np.clip((center_offset - (cascade_center_baseline or 0.0)) / cascade_shift_range, -1.0, 1.0))
                    shift_norm = smoothing_shift * previous_shift + (1 - smoothing_shift) * shift_raw

                    detection_source = "cascade"
                    no_detection_frames = 0
                else:
                    no_detection_frames += 1
                    if no_detection_frames >= 5:
                        open_norm = 0.0
                        pucker_norm = 0.0
                        wide_norm = 0.0
                        smile_norm = 0.0
                        tongue_norm = 0.0
                        shift_norm = 0.0

            if local_emotion_model is not None and now - emotion_last_inference >= EMOTION_PREDICTION_INTERVAL:
                faces_for_emotion = face_cascade.detectMultiScale(
                    equalized_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(120, 120),
                )
                if len(faces_for_emotion) == 0 and emotion_last_face is not None:
                    faces_for_emotion = np.array([emotion_last_face])

                if len(faces_for_emotion) > 0:
                    fx, fy, fw, fh = max(faces_for_emotion, key=lambda rect: rect[2] * rect[3])
                    emotion_last_face = (int(fx), int(fy), int(fw), int(fh))
                    face_roi = gray_frame[fy : fy + fh, fx : fx + fw]
                    if face_roi.size > 0:
                        try:
                            resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
                            normalized = resized.astype(np.float32) / 255.0
                            sample = normalized.reshape((1, 48, 48, 1))
                            predictions = local_emotion_model.predict(sample, verbose=0)  # type: ignore[attr-defined]
                            probs = np.array(predictions).reshape(-1)
                            if probs.size >= len(EMOTION_LABELS):
                                best_idx = int(np.argmax(probs))
                                score = float(np.clip(probs[best_idx], 0.0, 1.0))
                                label = EMOTION_LABELS.get(best_idx, "neutral")
                                if score < EMOTION_IDLE_THRESHOLD:
                                    emotion_score_state *= EMOTION_DECAY
                                else:
                                    if label == emotion_label_state:
                                        emotion_score_state = emotion_score_state * EMOTION_SMOOTHING + score * (1 - EMOTION_SMOOTHING)
                                    elif score > emotion_score_state + 0.12:
                                        emotion_label_state = label
                                        emotion_score_state = emotion_score_state * 0.35 + score * 0.65
                                    else:
                                        emotion_score_state = emotion_score_state * EMOTION_DECAY + score * (1 - EMOTION_DECAY) * 0.5
                                    emotion_label_state = label if emotion_score_state >= EMOTION_IDLE_THRESHOLD else "neutral"
                                emotion_score_state = float(np.clip(emotion_score_state, 0.0, 1.0))
                                emotion_last_inference = now
                            else:
                                emotion_score_state *= EMOTION_DECAY
                        except Exception as exc:  # pragma: no cover - defensive logging
                            if not EMOTION_MODEL_WARNING_EMITTED:
                                print(f"[EMOTION] Inference error: {exc}", file=sys.stderr)
                                EMOTION_MODEL_WARNING_EMITTED = True
                            print(f"[EMOTION] Inference error: {exc}", file=sys.stderr)
                            local_emotion_model = None
                            with emotion_model_lock:
                                emotion_model = None
                else:
                    emotion_score_state *= EMOTION_DECAY
                    if emotion_score_state < EMOTION_IDLE_THRESHOLD:
                        emotion_label_state = "neutral"

            detected = detection_source != "none"
            if not detected:
                open_norm = max(open_norm - 0.02, 0.0)
                pucker_norm = max(pucker_norm - 0.02, 0.0)
                wide_norm = max(wide_norm - 0.02, 0.0)
                smile_norm = max(smile_norm - 0.02, 0.0)
                tongue_norm = max(tongue_norm - 0.03, 0.0)
                shift_norm = shift_norm * 0.7

            confidence = float(np.clip(max(open_norm, pucker_norm, wide_norm, smile_norm, tongue_norm), 0.0, 1.0))

            with payload_lock:
                latest_payload = {
                    "ratio": float(round(open_norm, 4)),
                    "open": float(round(open_norm, 4)),
                    "pucker": float(round(pucker_norm, 4)),
                    "wide": float(round(wide_norm, 4)),
                    "smile": float(round(smile_norm, 4)),
                    "tongue": float(round(tongue_norm, 4)),
                    "shift": float(round(shift_norm, 4)),
                    "detected": detected,
                    "confidence": float(round(confidence, 4)),
                    "source": detection_source,
                    "horizontal": float(round(horizontal_ratio or 0.0, 4)),
                    "emotion": emotion_label_state,
                    "emotion_score": float(round(emotion_score_state, 4)),
                }

            previous_open = open_norm
            previous_pucker = pucker_norm
            previous_wide = wide_norm
            previous_smile = smile_norm
            previous_tongue = tongue_norm
            previous_shift = shift_norm

            if show_preview:
                preview_frame = frame.copy()
                if mouth_landmarks:
                    for (px, py) in mouth_landmarks:
                        cv2.circle(preview_frame, (px, py), 2, (0, 165, 255), -1)
                if inner_mouth_outline:
                    pts = np.array(inner_mouth_outline, dtype=np.int32)
                    cv2.polylines(preview_frame, [pts], True, (255, 140, 0), 1)
                if best_mouth_rect is not None:
                    (mx, my, mw, mh) = best_mouth_rect
                    color = (0, 200, 255) if detection_source == "mediapipe" else (0, 255, 0)
                    cv2.rectangle(preview_frame, (mx, my), (mx + mw, my + mh), color, 2)
                cv2.putText(
                    preview_frame,
                    f"Open:{open_norm:.2f} Puck:{pucker_norm:.2f} Wide:{wide_norm:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview_frame,
                    f"Smile:{smile_norm:.2f} Tongue:{tongue_norm:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    preview_frame,
                    f"Shift:{shift_norm:+.2f} Source:{detection_source}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    preview_frame,
                    f"Emotion:{emotion_label_state} ({emotion_score_state:.2f})",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (64, 255, 255),
                    2,
                )
                cv2.imshow("Mouth Detector Preview", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running = False

            time.sleep(BROADCAST_INTERVAL)
    finally:
        if face_mesh is not None:
            face_mesh.close()


async def register_client(websocket: websockets.WebSocketServerProtocol) -> None:
    with clients_lock:
        clients.add(websocket)
    with payload_lock:
        snapshot = json.dumps(latest_payload)
    await websocket.send(snapshot)


async def unregister_client(websocket: websockets.WebSocketServerProtocol) -> None:
    with clients_lock:
        clients.discard(websocket)


async def broadcast_loop() -> None:
    while running:
        await asyncio.sleep(BROADCAST_INTERVAL)
        with payload_lock:
            snapshot = json.dumps(latest_payload)
        to_remove: list[websockets.WebSocketServerProtocol] = []
        with clients_lock:
            active_clients = list(clients)
        if not active_clients:
            continue
        for websocket in active_clients:
            try:
                await websocket.send(snapshot)
            except websockets.ConnectionClosed:
                to_remove.append(websocket)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"WebSocket send error: {exc}")
                to_remove.append(websocket)
        if to_remove:
            with clients_lock:
                for websocket in to_remove:
                    clients.discard(websocket)


async def websocket_handler(websocket: websockets.WebSocketServerProtocol) -> None:
    await register_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        await unregister_client(websocket)


def shutdown_handler(*_args: object) -> None:
    global running
    running = False


def release_resources() -> None:
    global cap
    with clients_lock:
        clients_copy = list(clients)
        clients.clear()
    for websocket in clients_copy:
        try:
            asyncio.create_task(websocket.close(code=1001, reason="Server shutting down"))
        except RuntimeError:
            pass
    with cap_lock:
        if cap is not None:
            cap.release()
            cap = None
    cv2.destroyAllWindows()


async def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket mouth openness detector")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the camera to use (default: 0)")
    parser.add_argument("--list-cameras", action="store_true", help="List the first 5 camera indices and exit")
    parser.add_argument("--preview", action="store_true", help="Show a live preview window for debugging")
    parser.add_argument(
        "--disable-mediapipe",
        action="store_true",
        help="Disable MediaPipe FaceMesh and use Haar cascades only",
        help="Disable MediaPipe FaceMesh and use Haar cascades only (if mediapipe is not installed, this is default).",
    )
    args = parser.parse_args()

    use_mediapipe = not args.disable_mediapipe
    if not use_mediapipe:
        print("[SETUP] MediaPipe disabled via CLI flag; using Haar cascades only.")

    if args.list_cameras:
        print("Scanning first 5 camera indices...")
        for idx in range(5):
            cam = cv2.VideoCapture(idx)
            ok = cam.isOpened()
            label = "available" if ok else "unavailable"
            print(f"Camera {idx}: {label}")
            cam.release()
        return

    if not initialise_camera(args.camera_index):
        print(f"Could not open camera index {args.camera_index}. Trying to find another...")
        found_cam = False
        for i in range(5):
            if i == args.camera_index:
                continue
            if initialise_camera(i):
                print(f"Success! Using camera index {i}.")
                found_cam = True
                break
        if not found_cam:
            print("Unable to find any working camera. Please ensure it is connected and not in use by another application.")
            return
    else:
        print(f"Camera initialised using index {args.camera_index}.")

    global emotion_model
    global emotion_model, MEDIAPIPE_AVAILABLE
    with emotion_model_lock:
        if emotion_model is None:
            emotion_model = load_emotion_model()

    detection_thread = threading.Thread(
        target=detection_loop,
        args=(args.preview, use_mediapipe),
        daemon=True,
        daemon=True
    )
    detection_thread.start()

    serve_kwargs = {"ping_interval": None}
    if sys.platform != "win32":
        serve_kwargs["reuse_port"] = True

    async with websockets.serve(
        websocket_handler,
        "localhost",
        WEBSOCKET_PORT,
        **serve_kwargs,
    ):
        print(f"WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
        broadcast_task = asyncio.create_task(broadcast_loop())
        try:
            while running:
                await asyncio.sleep(0.1)
        finally:
            shutdown_handler()
            broadcast_task.cancel()
            with suppress(asyncio.CancelledError):
                await broadcast_task

    detection_thread.join(timeout=2)
    release_resources()


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown_handler)
    asyncio.run(main())
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        shutdown_handler()
