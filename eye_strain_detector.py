"""
Eye Strain and Emotion Detection Module

Ported from: Venkatesh020705/eye-strain-emotion-detection
Provides real-time detection of:
- Eye strain (based on EAR, blink frequency, and gaze variation)
- Facial emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- Fatigue alerts (fusion of eye strain + emotion indicators)

No Streamlit dependencies - designed for WebSocket integration
"""

import cv2
import numpy as np
import time
from collections import deque

# Try to import DeepFace for emotion detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Eye landmark indices for MediaPipe (6-point model per eye)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Valid emotion classes from DeepFace
VALID_EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']


def calculate_eye_aspect_ratio(landmarks, eye_indices, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Where p1, p2, p3, p4, p5, p6 are the eye landmarks.
    
    Returns:
        float: EAR value clamped between 0.0 and 1.0
    """
    try:
        # Extract coordinates from landmarks
        coords = [
            (int(landmarks[i].x * frame_width), int(landmarks[i].y * frame_height))
            for i in eye_indices
        ]
        
        # Convert to numpy array for easier computation
        points = np.array(coords, dtype=np.float32)
        
        # Vertical distances
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, ear))
    
    except Exception:
        return 0.0


class EnhancedBlinkDetector:
    """
    Advanced blink detector with temporal analysis and strain detection.
    
    Detects:
    - Individual blinks
    - Blink rate (blinks per minute)
    - Eye strain (prolonged eye closure, reduced blink frequency)
    - Strain counter for fusion with emotion detection
    """
    
    def __init__(self, ear_threshold=0.25, consecutive_frames=3, ear_smoothing_window=5):
        """
        Initialize blink detector.
        
        Args:
            ear_threshold: EAR threshold below which eye is considered closed
            consecutive_frames: Number of frames to consider for debouncing
            ear_smoothing_window: Window size for EAR smoothing
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.ear_smoothing_window = ear_smoothing_window
        
        # History tracking
        self.ear_history = deque(maxlen=30)  # Last 30 EAR values
        self.blink_times = deque(maxlen=60)  # Last 60 blink timestamps
        
        # State variables
        self.closed_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0
        self.strain_counter = 0
        self.current_ear = 0.0
        self.smoothed_ear = 0.0
    
    def update(self, ear, timestamp=None):
        """
        Update detector with new EAR value.
        
        Args:
            ear: Eye Aspect Ratio value
            timestamp: Time of frame (for accurate blink rate)
            
        Returns:
            dict: Contains blink_detected (bool), smoothed_ear (float), and other metrics
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store current EAR
        self.current_ear = ear
        self.ear_history.append(ear)
        
        # Compute smoothed EAR using moving average
        if len(self.ear_history) >= self.ear_smoothing_window:
            self.smoothed_ear = np.mean(
                list(self.ear_history)[-self.ear_smoothing_window:]
            )
        else:
            self.smoothed_ear = ear
        
        # Blink detection logic
        blink_detected = False
        
        if self.smoothed_ear < self.ear_threshold:
            # Eye is closed
            self.closed_frames += 1
            
            # Increment strain counter for prolonged closure (>10 frames = ~333ms at 30 FPS)
            if self.closed_frames > 10:
                self.strain_counter += 1
        else:
            # Eye is open
            if self.closed_frames >= self.consecutive_frames:
                # Debouncing: minimum 300ms between blinks
                if timestamp - self.last_blink_time > 0.3:
                    self.blink_count += 1
                    self.last_blink_time = timestamp
                    self.blink_times.append(timestamp)
                    blink_detected = True
            
            self.closed_frames = 0
            
            # Decrease strain counter when eyes are open normally
            self.strain_counter = max(0, self.strain_counter - 1)
        
        return {
            'blink_detected': blink_detected,
            'smoothed_ear': self.smoothed_ear,
            'current_ear': ear,
            'strain_counter': self.strain_counter
        }
    
    def get_stats(self):
        """
        Get current detector statistics.
        
        Returns:
            dict: Statistics including blink_count, strain_level, avg_ear, blink_rate
        """
        # Calculate blink rate (blinks per minute)
        blink_rate = 0.0
        if len(self.blink_times) > 1:
            time_span = self.blink_times[-1] - self.blink_times[0]
            if time_span > 0:
                blink_rate = (len(self.blink_times) / time_span) * 60
        
        avg_ear = np.mean(list(self.ear_history)) if self.ear_history else 0.0
        
        return {
            'blink_count': self.blink_count,
            'strain_level': self.strain_counter,
            'avg_ear': avg_ear,
            'current_ear': self.current_ear,
            'smoothed_ear': self.smoothed_ear,
            'blink_rate': blink_rate
        }
    
    def reset(self):
        """Reset detector for new session."""
        self.ear_history.clear()
        self.blink_times.clear()
        self.closed_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0
        self.strain_counter = 0
        self.current_ear = 0.0
        self.smoothed_ear = 0.0


class EmotionDetector:
    """
    Facial emotion detection using DeepFace.
    
    Detects: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
    Returns confidence scores and dominant emotion.
    """
    
    def __init__(self, detector_backend='opencv', enforce_detection=False):
        """
        Initialize emotion detector.
        
        Args:
            detector_backend: Backend to use ('opencv', 'ssd', 'dlib', etc.)
            enforce_detection: Whether to enforce face detection
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.available = DEEPFACE_AVAILABLE
        self.last_emotion = 'neutral'
        self.last_confidence = 0.0
        self.emotion_history = deque(maxlen=10)  # Last 10 detections
    
    def detect(self, frame):
        """
        Detect facial emotion in frame.
        
        Args:
            frame: Input frame in RGB format
            
        Returns:
            dict: Contains emotion (str), confidence (float), and error (str if failed)
        """
        if not self.available:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': 'DeepFace not available'
            }
        
        try:
            # Convert RGB to BGR for DeepFace (OpenCV format)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Run DeepFace analysis
            result = DeepFace.analyze(
                frame_bgr,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                result = result[0]
            
            # Extract emotions and dominant emotion
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            
            # Get confidence (convert percentage to 0-1 range)
            confidence = emotions[dominant_emotion] / 100.0
            
            # Normalize emotion name
            emotion_name = dominant_emotion.lower()
            if emotion_name not in VALID_EMOTIONS:
                emotion_name = 'neutral'
            
            self.last_emotion = emotion_name
            self.last_confidence = confidence
            self.emotion_history.append(emotion_name)
            
            return {
                'emotion': emotion_name,
                'confidence': confidence,
                'all_emotions': {k: v/100.0 for k, v in emotions.items()},
                'error': None
            }
        
        except Exception as e:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e)[:50]
            }
    
    def get_dominant_emotion_history(self):
        """
        Get most common emotion from recent history.
        
        Returns:
            str: Most common emotion from last 10 detections
        """
        if not self.emotion_history:
            return 'neutral'
        
        # Count occurrences
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return most common
        return max(emotion_counts, key=emotion_counts.get)


class FatigueDetector:
    """
    Comprehensive fatigue detection combining:
    - Eye strain (EAR-based)
    - Emotion indicators (tired/sad/neutral patterns)
    - Blink frequency (reduced blinking = fatigue)
    - Fusion logic for accurate alerts
    
    Uses personalized calibration thresholds.
    """
    
    def __init__(self, 
                 strain_threshold=50,
                 alert_cooldown=30,
                 emotion_weight=0.4,
                 strain_weight=0.6):
        """
        Initialize fatigue detector.
        
        Args:
            strain_threshold: Threshold for strain level to trigger alert
            alert_cooldown: Minimum seconds between alerts
            emotion_weight: Weight of emotion in fusion (0-1)
            strain_weight: Weight of strain in fusion (0-1)
        """
        self.strain_threshold = strain_threshold
        self.alert_cooldown = alert_cooldown
        self.emotion_weight = emotion_weight
        self.strain_weight = strain_weight
        
        # Initialize sub-detectors
        self.blink_detector = EnhancedBlinkDetector()
        self.emotion_detector = EmotionDetector()
        
        # Alert tracking
        self.last_alert_time = 0
        self.alert_count = 0
        self.fatigue_level = 0.0
        
        # Calibration data
        self.calibrated = False
        self.normal_blink_rate = 17.0  # Default: 17 blinks/minute
        self.normal_ear = 0.35  # Default normal EAR
    
    def set_calibration(self, normal_blink_rate, normal_ear, custom_thresholds=None):
        """
        Set personalized calibration parameters.
        
        Args:
            normal_blink_rate: Normal blink rate for user (blinks/minute)
            normal_ear: Normal EAR value for user
            custom_thresholds: Dict with 'strain_threshold', 'alert_cooldown'
        """
        self.normal_blink_rate = max(5, normal_blink_rate)  # Minimum 5 bpm
        self.normal_ear = max(0.15, normal_ear)  # Minimum 0.15
        
        if custom_thresholds:
            self.strain_threshold = custom_thresholds.get(
                'strain_threshold', self.strain_threshold
            )
            self.alert_cooldown = custom_thresholds.get(
                'alert_cooldown', self.alert_cooldown
            )
        
        self.calibrated = True
    
    def process_frame(self, landmarks, frame, frame_width, frame_height):
        """
        Process single frame for fatigue detection.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame: RGB frame for emotion detection
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            dict: Detection results including fatigue_level, alert status, metrics
        """
        current_time = time.time()
        
        # Calculate EAR for both eyes
        left_ear = calculate_eye_aspect_ratio(
            landmarks, LEFT_EYE_INDICES, frame_width, frame_height
        )
        right_ear = calculate_eye_aspect_ratio(
            landmarks, RIGHT_EYE_INDICES, frame_width, frame_height
        )
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Update blink detector
        blink_update = self.blink_detector.update(avg_ear, current_time)
        blink_stats = self.blink_detector.get_stats()
        
        # Detect emotion (every few frames to save computation)
        emotion_result = self.emotion_detector.detect(frame)
        emotion = emotion_result['emotion']
        emotion_confidence = emotion_result['confidence']
        
        # Calculate strain score (0-100)
        # Based on: low EAR, high strain counter, reduced blink rate
        strain_component = min(100, self.blink_detector.strain_counter * 2)
        
        # Reduced blink rate component
        blink_rate_ratio = max(0, 1.0 - (blink_stats['blink_rate'] / self.normal_blink_rate))
        blink_component = blink_rate_ratio * 50  # 0-50 points
        
        # EAR component
        ear_ratio = max(0, 1.0 - (avg_ear / self.normal_ear))
        ear_component = ear_ratio * 50  # 0-50 points
        
        strain_level = min(100, (strain_component + blink_component + ear_component) / 1.5)
        
        # Calculate emotion fatigue indicator (0-100)
        # Tired/sad/neutral emotions indicate fatigue
        emotion_fatigue = 0.0
        if emotion in ['sad', 'neutral']:
            emotion_fatigue = emotion_confidence * 100
        elif emotion == 'fear':
            emotion_fatigue = emotion_confidence * 60
        else:
            emotion_fatigue = 0
        
        # Fusion logic: combine strain and emotion
        # Fatigue alert only when BOTH indicators align
        self.fatigue_level = (
            self.strain_weight * strain_level + 
            self.emotion_weight * emotion_fatigue
        )
        
        # Determine if alert should be raised
        should_alert = False
        alert_reason = None
        
        if strain_level >= self.strain_threshold:
            # Both strain and emotion indicators should align
            if emotion_fatigue >= 30:  # Emotion component threshold
                if current_time - self.last_alert_time > self.alert_cooldown:
                    should_alert = True
                    alert_reason = f"Fatigue detected (strain: {strain_level:.0f}, emotion: {emotion})"
                    self.last_alert_time = current_time
                    self.alert_count += 1
        
        return {
            'fatigue_level': self.fatigue_level,
            'fatigue_alert': should_alert,
            'alert_reason': alert_reason,
            'alert_count': self.alert_count,
            'strain_level': strain_level,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'emotion_fatigue': emotion_fatigue,
            'blink_rate': blink_stats['blink_rate'],
            'blink_count': blink_stats['blink_count'],
            'avg_ear': avg_ear,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'blink_detected': blink_update['blink_detected'],
            'detection_error': emotion_result.get('error')
        }
    
    def get_session_summary(self):
        """
        Get summary statistics for current session.
        
        Returns:
            dict: Session summary including averages and totals
        """
        stats = self.blink_detector.get_stats()
        
        return {
            'total_blinks': stats['blink_count'],
            'avg_ear': stats['avg_ear'],
            'blink_rate': stats['blink_rate'],
            'total_fatigue_alerts': self.alert_count,
            'avg_fatigue_level': self.fatigue_level,
            'dominant_emotion': self.emotion_detector.get_dominant_emotion_history(),
            'emotion_available': self.emotion_detector.available,
            'calibrated': self.calibrated
        }
    
    def reset(self):
        """Reset detector for new session."""
        self.blink_detector.reset()
        self.last_alert_time = 0
        self.alert_count = 0
        self.fatigue_level = 0.0
