import sys
import asyncio
import websockets
import json
import cv2
import numpy as np
import argparse
import threading
import logging
import os

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# Enhanced detectors
try:
    from eye_strain_detector import FatigueDetector, EmotionDetector
    FATIGUE_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f'[WARNING] Eye strain detector not available: {e}')
    FATIGUE_DETECTOR_AVAILABLE = False


# ---# Constants for mouth detection (adjust as needed)
# MOUTH_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_mouth.xml'
MOUTH_OPEN_THRESHOLD_FACTOR = 5.0  # Multiplier for the raw ratio to make it more sensitive
# Constants and Configuration ---
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Face detection model
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
if face_cascade.empty():
    raise FileNotFoundError("Face cascade classifier failed to load. Please check the path to 'haarcascade_frontalface_default.xml'.")

# --- Emotion Recognition Model Loading ---
emotion_model = None
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

try:
    # Load the model from JSON file - use absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(script_dir), 'Face-detection-model')
    json_path = os.path.join(model_dir, 'emotiondetector.json')
    weights_path = os.path.join(model_dir, 'emotiondetector.h5')
    
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        emotion_model = model_from_json(loaded_model_json)

    # Load weights and compile the model
    if emotion_model:
        emotion_model.load_weights(weights_path)
        print("[EMOTION] Loaded emotion recognition model successfully.")
except Exception as e:
    print(f"[EMOTION] Error loading emotion recognition model: {e}")


# --- Fatigue and Emotion Detection ---
fatigue_detector = None
emotion_detector_enhanced = None

if FATIGUE_DETECTOR_AVAILABLE:
    try:
        fatigue_detector = FatigueDetector(strain_threshold=50, alert_cooldown=30)
        emotion_detector_enhanced = EmotionDetector(detector_backend='opencv', enforce_detection=False)
        print('[FATIGUE] Eye strain and emotion detectors initialized successfully.')
    except Exception as e:
        print(f'[FATIGUE] Error initializing detectors: {e}')
        fatigue_detector = None
        emotion_detector_enhanced = None
else:
    print('[FATIGUE] Eye strain detector module not available - skipping initialization')

# --- Global State ---
# Global variable to hold the latest mouth data and a lock for thread safety
latest_data = {
    "ratio": 0.0,
    "detected": False,
    "confidence": 0.0,
    "emotion": "Neutral",
    "emotion_score": 0.0,
    "type": "emotion_mouth"
}
data_lock = threading.Lock()
clients = set()

# --- Image Processing and Detection ---
def detect_mouth_and_emotion(frame):
    """
    Detects face, mouth openness, and emotion from a single video frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    new_data = {
        "ratio": 0.0,
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
        "glare_level": 0.0,
        "has_glasses": False,
        "pursed_lips": 0.0,
        "smile_expression": 0.0,
        "frown_expression": 0.0,
        "lip_color_intensity": 0.0,
        "source": "haar",
        "type": "emotion_mouth"
    }

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Use the first detected face
        
        # --- Mouth Detection ---
        # A simple heuristic for mouth openness based on face height
        # This part can be replaced with a more sophisticated mouth landmark detector
        mouth_roi_gray = gray[y + h//2 : y + h, x : x + w]
        if mouth_roi_gray.size > 0:
            _, thresh = cv2.threshold(mouth_roi_gray, 120, 255, cv2.THRESH_BINARY_INV)
            open_pixels = cv2.countNonZero(thresh)
            total_pixels = mouth_roi_gray.size
            ratio = open_pixels / total_pixels if total_pixels > 0 else 0
            mouth_openness = min(ratio * 5.0, 1.0) # Amplify for visibility
            
            new_data["ratio"] = mouth_openness
            new_data["open"] = mouth_openness  # Main mouth openness value
            new_data["detected"] = True
            new_data["confidence"] = 0.85
            
            # Basic heuristic values (can be enhanced with more sophisticated detection)
            # For now, derive simple values from mouth dimensions
            new_data["wide"] = min((w / h) * 0.3, 1.0) if h > 0 else 0  # Width-to-height ratio
            new_data["smile"] = 0  # Would need corner detection for accurate smile
            new_data["pucker"] = 0  # Would need lip shape analysis
            new_data["tongue"] = 0  # Would need internal mouth analysis
            new_data["shift"] = (x + w/2 - frame.shape[1]/2) / (frame.shape[1]/2)  # Horizontal position

        # --- Emotion Detection ---
        if emotion_model:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi_gray, (48, 48))
            image_pixels = img_to_array(face_roi_resized)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255.0

            predictions = emotion_model.predict(image_pixels, verbose=0)
            max_index = int(np.argmax(predictions))
            
            emotion_label = emotion_labels[max_index]
            emotion_score = float(np.max(predictions))
            
            new_data["emotion"] = emotion_label
            new_data["emotion_score"] = emotion_score
            
            # Map emotions to facial expressions
            if emotion_label == 'Happy' and emotion_score > 0.5:
                new_data["smile"] = emotion_score
                new_data["smile_expression"] = emotion_score
            elif emotion_label == 'Sad' and emotion_score > 0.5:
                new_data["frown_expression"] = emotion_score
            elif emotion_label == 'Surprise' and emotion_score > 0.5:
                new_data["wide"] = max(new_data["wide"], emotion_score * 0.7)

    return new_data

# --- WebSocket Server Logic ---
async def handler(websocket, path):
    """
    Main WebSocket handler. Broadcasts data from the camera thread.
    """
    global latest_data
    print("[SERVER] Client connected.")
    clients.add(websocket)
    try:
        # This part is now for broadcasting, the main loop is in the camera thread
        while True:
            with data_lock:
                message = json.dumps(latest_data)
            await websocket.send(message)
            await asyncio.sleep(1/45)  # Broadcast at ~45 FPS
    except websockets.exceptions.ConnectionClosed:
        print("[SERVER] Client disconnected.")
    finally:
        clients.remove(websocket)

async def broadcast(message):
    if clients:
        await asyncio.wait([client.send(message) for client in clients])

def camera_thread(camera_index, preview):
    """
    Thread to capture from camera and update global data.
    """
    global latest_data
    print(f"[CAMERA] Starting camera thread with index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}. Please check the camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[CAMERA] Camera {camera_index} opened successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA] Failed to grab frame.")
            break

        new_data = detect_mouth_and_emotion(frame)

        with data_lock:
            latest_data = new_data

        if preview:
            # Draw debug info on the preview frame
            cv2.putText(frame, f"Ratio: {new_data['ratio']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {new_data['emotion']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Rocko Backend Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if preview:
        cv2.destroyAllWindows()
    print("[CAMERA] Camera thread stopped.")

def list_available_cameras():
    """
    Lists all available camera devices.
    """
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

    # Start the camera capture in a separate thread
    cam_thread = threading.Thread(target=camera_thread, args=(camera_index, preview), daemon=True)
    cam_thread.start()

    # Start the WebSocket server
    print(f"[SERVER] Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port, max_size=1_000_000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocko WebSocket Server for Mouth and Emotion Tracking")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the camera to use for the backend.")
    parser.add_argument("--preview", action="store_true", help="Show a preview window for the backend camera.")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the WebSocket server.")
    parser.add_argument("--port", type=int, default=6789, help="Port for the WebSocket server.")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera devices and exit.")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port, args.camera_index, args.preview, args.list_cameras))
    except (KeyboardInterrupt, RuntimeError):
        print("\n[SERVER] Shutting down server.")
