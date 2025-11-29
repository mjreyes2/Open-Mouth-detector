import sys
import os

# Get the absolute path to the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the site-packages directory of the correct virtual environment
venv_path = os.path.join(script_dir, '.venv311', 'Lib', 'site-packages')

# Add the site-packages directory to the Python path
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

import asyncio
import websockets
import json
import cv2
import numpy as np
import argparse
import threading
import logging
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
from typing import TypedDict, Union, List, Dict

# ---# Constants and Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Emotion Recognition Model Loading ---
emotion_model = None
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(script_dir), 'Face-detection-model')
    json_path = os.path.join(model_dir, 'emotiondetector.json')
    weights_path = os.path.join(model_dir, 'emotiondetector.h5')
    
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        emotion_model = model_from_json(loaded_model_json)

    if emotion_model:
        emotion_model.load_weights(weights_path)
        print("[EMOTION] Loaded emotion recognition model successfully.")
except Exception as e:
    print(f"[EMOTION] Error loading emotion recognition model: {e}")

# --- Global State ---
latest_data = {
    "open": 0.0,
    "pucker": 0.0,
    "wide": 0.0,
    "smile": 0.0,
    "tongue": 0.0,
    "shift": 0.0,
    "detected": False,
    "confidence": 0.0,
    "emotion": "Neutral",
    "emotion_score": 0.0,
    "gaze_x": 0.0,
    "gaze_y": 0.0,
    "head_pose_pitch": 0.0,
    "head_pose_yaw": 0.0,
    "head_pose_roll": 0.0,
    "glare_level": 0.0,
    "has_glasses": False,
    "source": "mediapipe",
    "type": "face_tracking",
    "blink_left": 0.0,
    "blink_right": 0.0,
    "blink": 0.0
}
data_lock = threading.Lock()
clients = set()

# Lip landmark indices for MediaPipe Face Mesh (468 landmarks)
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17

# Eye landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Define a type hint for the data structure for clarity and improved linting
class TrackingData(TypedDict, total=False):
    open: float
    pucker: float
    wide: float
    smile: float
    tongue: float
    shift: float
    ratio: float
    detected: bool
    confidence: float
    emotion: str
    emotion_score: float
    gaze_x: float
    gaze_y: float
    head_pose_pitch: float
    head_pose_yaw: float
    head_pose_roll: float
    glare_level: float
    has_glasses: bool
    pursed_lips: float
    smile_expression: float
    frown_expression: float
    lip_color_intensity: float
    blink_left: float
    blink_right: float
    blink: float

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two landmarks."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def convert_numpy_types(obj):
    """Recursively convert all numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def detect_face_and_mouth_mediapipe(frame, face_mesh) -> TrackingData:
    """
    Enhanced face detection using MediaPipe Face Mesh with detailed mouth tracking.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    new_data: TrackingData = {
        "open": 0.0,
        "pucker": 0.0,
        "wide": 0.0,
        "smile": 0.0,
        "tongue": 0.0,
        "shift": 0.0,
        "detected": False,
        "confidence": 0.0,
        "emotion": "Neutral",
        "emotion_score": 0.0,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
        "head_pose_pitch": 0.0,
        "head_pose_yaw": 0.0,
        "head_pose_roll": 0.0,
        "glare_level": 0.0,
        "has_glasses": False,
        "pursed_lips": 0.0,
        "smile_expression": 0.0,
        "frown_expression": 0.0,
        "lip_color_intensity": 0.0,
        "blink_left": 0.0,
        "blink_right": 0.0,
        "blink": 0.0,
    }
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape
            
            new_data["detected"] = True
            new_data["confidence"] = 0.95
            
            # === MOUTH OPENNESS (Vertical) - ULTRA SENSITIVE ===
            upper_lip = landmarks[UPPER_LIP_TOP]
            lower_lip = landmarks[LOWER_LIP_BOTTOM]
            mouth_height = calculate_distance(upper_lip, lower_lip)
            # ULTRA BOOST: multiply by 35 for maximum sensitivity to subtle movements
            mouth_open_normalized = float(min(mouth_height * 35.0, 1.0))
            new_data["open"] = mouth_open_normalized
            new_data["ratio"] = mouth_open_normalized  # Compatibility alias
            
            # === MOUTH WIDTH (Horizontal) ===
            left_corner = landmarks[LEFT_LIP_CORNER]
            right_corner = landmarks[RIGHT_LIP_CORNER]
            mouth_width = calculate_distance(left_corner, right_corner)
            
            # Average mouth width reference (wider mouth = smile or wide expression)
            mouth_width_normalized = float(min(mouth_width * 15.0, 1.0))  # Increased from 12
            new_data["wide"] = mouth_width_normalized
            
            # === PUCKER DETECTION (lips pushed forward) ===
            # Calculate average upper/lower lip positions
            upper_lip_points = [landmarks[i] for i in UPPER_LIP_INDICES]
            lower_lip_points = [landmarks[i] for i in LOWER_LIP_INDICES]
            
            avg_upper_z = sum(p.z for p in upper_lip_points) / len(upper_lip_points)
            avg_lower_z = sum(p.z for p in lower_lip_points) / len(lower_lip_points)
            
            # Pucker is when lips move forward (negative z)
            pucker_value = max(0, -(avg_upper_z + avg_lower_z) * 10.0)  # Increased from 8
            new_data["pucker"] = float(min(pucker_value, 1.0))
            
            # === PURSED LIPS (tight compression) ===
            # When mouth width is small relative to openness
            if mouth_width_normalized > 0.1:
                pursed_ratio = mouth_open_normalized / mouth_width_normalized
                new_data["pursed_lips"] = float(min(pursed_ratio * 2.5, 1.0))
            
            # === SMILE DETECTION (corners up) ===
            nose_tip = landmarks[1]
            left_corner_height = left_corner.y - nose_tip.y
            right_corner_height = right_corner.y - nose_tip.y
            
            # Negative value means corners are above nose (smile)
            smile_lift = max(0, -(left_corner_height + right_corner_height) * 6.0)  # Increased from 5
            new_data["smile"] = float(min(smile_lift, 1.0))
            new_data["smile_expression"] = new_data["smile"]
            
            # === FROWN DETECTION (corners down) ===
            frown_drop = max(0, (left_corner_height + right_corner_height) * 6.0)  # Increased from 5
            new_data["frown_expression"] = float(min(frown_drop, 1.0))
            
            # === HORIZONTAL SHIFT (face position) ===
            face_center_x = (left_corner.x + right_corner.x) / 2
            new_data["shift"] = float((face_center_x - 0.5) * 2.0)  # -1 to 1 range
            
            # === GAZE TRACKING (eye direction) ===
            # Use iris landmarks if available (468-477)
            if len(landmarks) > 470:
                left_iris = landmarks[LEFT_IRIS_CENTER]
                right_iris = landmarks[RIGHT_IRIS_CENTER]
                
                # Eye centers
                left_eye_center = landmarks[33]
                right_eye_center = landmarks[263]
                
                # Calculate gaze direction with ULTRA sensitivity for noticeable movement
                left_gaze_x = (left_iris.x - left_eye_center.x) * 25.0  # ULTRA sensitive
                left_gaze_y = (left_iris.y - left_eye_center.y) * 25.0
                right_gaze_x = (right_iris.x - right_eye_center.x) * 25.0
                right_gaze_y = (right_iris.y - right_eye_center.y) * 25.0
                
                # Average both eyes
                new_data["gaze_x"] = float((left_gaze_x + right_gaze_x) / 2.0)
                new_data["gaze_y"] = float((left_gaze_y + right_gaze_y) / 2.0)
            
            # === BLINK DETECTION ===
            # Calculate Eye Aspect Ratio (EAR) for blink detection
            def eye_aspect_ratio(eye_landmarks):
                # Vertical eye landmarks (top and bottom)
                v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])  # top to bottom left
                v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])  # top to bottom right
                # Horizontal eye landmark (left to right corner)
                h = calculate_distance(eye_landmarks[0], eye_landmarks[3])  # left to right corner
                # EAR formula
                ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
                return ear
            
            # Left eye EAR calculation
            left_eye_points = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]  # Left eye landmarks
            if len(left_eye_points) >= 6:
                left_ear = eye_aspect_ratio(left_eye_points)
                # Normalize EAR (typical open eye EAR is ~0.2-0.4, closed eye is ~0.05-0.15)
                # Convert to blink strength (0 = open, 1 = closed/blink)
                left_blink = max(0, min(1, (0.25 - left_ear) * 6.0))
                new_data["blink_left"] = float(left_blink)
            
            # Right eye EAR calculation  
            right_eye_points = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]  # Right eye landmarks
            if len(right_eye_points) >= 6:
                right_ear = eye_aspect_ratio(right_eye_points)
                right_blink = max(0, min(1, (0.25 - right_ear) * 6.0))
                new_data["blink_right"] = float(right_blink)
            
            # Overall blink (average of both eyes)
            new_data["blink"] = float((new_data["blink_left"] + new_data["blink_right"]) / 2.0)
            
            # === HEAD POSE ESTIMATION ===
            # Using key facial landmarks for 3D pose estimation
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            left_ear = landmarks[234]
            right_ear = landmarks[454]
            
            # Pitch (up/down tilt)
            pitch = (nose_tip.y - chin.y) * 5.0
            new_data["head_pose_pitch"] = float(np.clip(pitch, -1.0, 1.0))
            
            # Yaw (left/right rotation)
            yaw = (right_ear.z - left_ear.z) * 5.0
            new_data["head_pose_yaw"] = float(np.clip(yaw, -1.0, 1.0))
            
            # Roll (tilt sideways)
            roll = (right_eye.y - left_eye.y) * 5.0
            new_data["head_pose_roll"] = float(np.clip(roll, -1.0, 1.0))
            
            # === GLASSES DETECTION ===
            # Check for brightness anomalies around eyes (reflection from glasses)
            left_eye_roi = frame[int(left_eye.y*h)-10:int(left_eye.y*h)+10, 
                                 int(left_eye.x*w)-10:int(left_eye.x*w)+10]
            right_eye_roi = frame[int(right_eye.y*h)-10:int(right_eye.y*h)+10,
                                  int(right_eye.x*w)-10:int(right_eye.x*w)+10]
            
            if left_eye_roi.size > 0 and right_eye_roi.size > 0:
                left_brightness = float(np.mean(left_eye_roi))
                right_brightness = float(np.mean(right_eye_roi))
                avg_brightness = (left_brightness + right_brightness) / 2
                
                # High brightness suggests glass reflections
                new_data["glare_level"] = float(min(avg_brightness / 255.0, 1.0))
                new_data["has_glasses"] = bool(new_data["glare_level"] > 0.4)
            
            # === TONGUE DETECTION (experimental) ===
            # Check if there's movement in the inner mouth area
            inner_mouth = landmarks[13]  # Inside lower lip
            if inner_mouth.z < lower_lip.z - 0.01:
                new_data["tongue"] = float(min((lower_lip.z - inner_mouth.z) * 18.0, 1.0))  # Increased from 15
            
            # === EMOTION DETECTION ===
            if emotion_model:
                x_min = int(min(l.x for l in landmarks) * w)
                y_min = int(min(l.y for l in landmarks) * h)
                x_max = int(max(l.x for l in landmarks) * w)
                y_max = int(max(l.y for l in landmarks) * h)
                
                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (48, 48))
                    image_pixels = img_to_array(resized_face)
                    image_pixels = np.expand_dims(image_pixels, axis=0)
                    image_pixels /= 255.0
                    
                    predictions = emotion_model.predict(image_pixels, verbose=0)
                    max_index = int(np.argmax(predictions))
                    
                    new_data["emotion"] = str(emotion_labels[max_index])
                    new_data["emotion_score"] = float(np.max(predictions))
        
        # CRITICAL: Convert ALL numpy types to Python native types recursively
        new_data = convert_numpy_types(new_data)

    return new_data

# --- WebSocket Server Logic ---
async def handler(websocket):
    """Main WebSocket handler."""
    print("[SERVER] Client connected.")
    clients.add(websocket)
    try:
        while True:
            with data_lock:
                message = json.dumps(latest_data)
            await websocket.send(message)
            await asyncio.sleep(1/60)  # 60 FPS
    except websockets.exceptions.ConnectionClosed:
        print("[SERVER] Client disconnected.")
    finally:
        clients.remove(websocket)

def camera_thread(camera_index, preview):
    """Thread to capture from camera and update global data."""
    global latest_data
    print(f"[CAMERA] Starting MediaPipe camera thread with index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # Set a higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            new_data = detect_face_and_mouth_mediapipe(frame, face_mesh)
            
            with data_lock:
                latest_data = new_data.copy()

            if preview:
                # Draw the data on the frame
                cv2.putText(frame, f"Open: {new_data.get('open', 0):.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Wide: {new_data.get('wide', 0):.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Smile: {new_data.get('smile', 0):.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze: ({new_data.get('gaze_x', 0):.2f}, {new_data.get('gaze_y', 0):.2f})",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {new_data.get('emotion', 'N/A')}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Blink: L{new_data.get('blink_left', 0):.2f} R{new_data.get('blink_right', 0):.2f} ({new_data.get('blink', 0):.2f})",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Camera Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    if preview:
        cv2.destroyAllWindows()
    print("[CAMERA] Camera thread stopped.")

def list_available_cameras():
    """Lists all available camera devices."""
    print("Searching for available cameras...")
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    if not arr:
        print("No cameras found.")
    else:
        print("Available camera indices:", arr)

# --- Main Execution ---
async def main(host, port, camera_index, preview, list_cameras_flag):
    if list_cameras_flag:
        list_available_cameras()
        return

    cam_thread = threading.Thread(target=camera_thread, args=(camera_index, preview), daemon=True)
    cam_thread.start()

    print(f"[SERVER] Starting MediaPipe WebSocket server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port, max_size=1_000_000):
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocko MediaPipe WebSocket Server")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index.")
    parser.add_argument("--preview", action="store_true", help="Show preview window.")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket host.")
    parser.add_argument("--port", type=int, default=6789, help="WebSocket port.")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras.")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port, args.camera_index, args.preview, args.list_cameras))
    except (KeyboardInterrupt, RuntimeError):
        print("\n[SERVER] Shutting down server.")
