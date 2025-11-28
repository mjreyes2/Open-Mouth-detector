# Real-time Mouth Recognition with OpenCV

This project implements real-time mouth recognition using OpenCV. It detects mouths in a video stream from the default camera and applies visual markers on the edges of the detected mouth. It also applies a filter effect to differentiate when the mouth is opened wider than a specified threshold.

## Usage

1. Make sure you have Python and OpenCV installed.
2. Clone this repository.
3. Install the required dependencies using the command `pip install -r requirements.txt`.
4. To view the standalone OpenCV demo, run `python main.py`. A video window will open showing the live camera stream with mouth detection. Press `q` to quit.
5. To stream mouth metrics to the web app, run `python mouth_websocket.py`. This exposes a WebSocket server on `ws://localhost:6789` that now publishes a richer bundle of normalized cues—`open`, `pucker`, `wide`, `smile`, `tongue`, and a signed `shift` value—alongside the legacy `ratio`. The detector defaults to MediaPipe FaceMesh for precise lip landmarks, with the classic Haar-cascade pipeline kept as a fallback. When FaceMesh is unavailable, the fallback now also taps OpenCV’s built-in smile cascade to keep the `smile` channel responsive instead of only widening-driven heuristics. An adaptive baseline keeps recalibrating to subtle lip movements (like pursing) so the output stays responsive. The tongue flag uses simple colour heuristics, so expect best results in even lighting.
	- Use the repository's Python 3.11 virtual environment (`.venv311`) if you have it set up: `.\.venv311\Scripts\python.exe mouth_websocket.py`. If TensorFlow/Keras is available and the `Face-detection-model` assets (`emotiondetector.json` and `emotiondetector.h5`) are present in the repository root, the server also emits an `emotion` label with an `emotion_score` confidence that the front-end can use for facial cues.
	- Use the repository's Python 3.11 virtual environment (`.venv311`) if you have it set up: `.\.venv311\Scripts\python.exe mouth_websocket.py`.
	- Optional: Install TensorFlow/Keras in that environment and keep the `Face-detection-model` assets (`emotiondetector.json` and `emotiondetector.h5`) in the repo root to enable the `emotion` label with its `emotion_score` confidence in the WebSocket payload.
	- Add `--preview` if you want to see the detection boxes while calibrating.
	- Use `--disable-mediapipe` to fall back to the legacy cascade-only detector if FaceMesh cannot initialise on your hardware.
6. If the wrong camera is selected (or nothing shows up), list available devices with `python mouth_websocket.py --list-cameras` and then relaunch with a specific index, for example `python mouth_websocket.py --camera-index 1`. The server now reuses the same port even after quick restarts.
7. Start your browser experience (for example, `speech_mouth.html` in the related LipSync.js project) once the server reports that it is running. The page will connect automatically and mimic your mouth movements. Watch the live “Camera open / pucker / wide / smile / tongue / shift …” badge for stats (it also reports whether MediaPipe or the cascade is driving the animation and shows when a lock is acquired). The `shift` channel now drives a lateral offset on the web component, so side-mouth poses show up immediately when you purse to one corner.

## Requirements

- Python 3.7 or higher (3.11 recommended for the provided virtual environment)
- `opencv-python` 4.8 or compatible version
- `numpy<2`
- `websockets`
- `mediapipe`

## Customization

- Adjust the `frame_width` and `frame_height` variables in `main.py` to set the desired video frame size.
- Customize the `mouth_open_threshold` value in `main.py` to change the threshold for detecting an open mouth.
- Tweak the constants in `mouth_websocket.py` (for example `min_ratio`, `ratio_range`, and the mapping in your front-end) if you need different sensitivity for the WebSocket feed.

## Notes

- The `haarcascade_mouth.xml` file is required for mouth detection. Please ensure that you have a valid Haar cascade XML file for mouth detection and update the code accordingly. The smile fallback uses OpenCV’s bundled `haarcascade_smile.xml`; no extra download is needed if your OpenCV package is intact.
- The accuracy of mouth detection may vary depending on factors such as lighting conditions and the quality of the input video stream.

Feel free to explore and modify the code to suit your specific needs!

