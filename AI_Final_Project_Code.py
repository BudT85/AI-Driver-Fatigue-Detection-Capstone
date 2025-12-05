"""
dlib_driver_monitor.py
Driver drowsiness detector & lazy-eye screening using dlib (no MediaPipe).
- Uses dlib face detector + 68-point landmark predictor.
- EAR (eye aspect ratio) for drowsiness detection.
- Simple lazy-eye asymmetry screening based on eye centers.
- Alert manager: lights, haptic (MQTT), escalating audible alarm (simpleaudio).

NOT a medical device. Lazy-eye check is only a heuristic screening.
"""
import time
import threading
import math
from collections import deque
import argparse
import os
import json

# Core libs (install via pip)
import dlib
import numpy as np
from imutils import face_utils

# Optional: OpenCV for capture/display. If unavailable, the script still runs detection if provided a video file.
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# Optional: MQTT to send actuations to microcontroller (seat/wheel vibration, LEDs).
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except Exception:
    MQTT_AVAILABLE = False

# Optional: simpleaudio for tones
try:
    import simpleaudio as sa
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

# ---------- Configuration ----------
EAR_THRESHOLD = 0.22         # adjust per-person (lower -> more sensitive)
CONSEC_FRAMES = 20           # frames below threshold to trigger drowsiness
STRABISMUS_ASYM_THRESHOLD = 0.07  # heuristic asymmetry threshold
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC_VIBRATION = "car/alert/vibration"
MQTT_TOPIC_LIGHTS = "car/alert/lights"

# Landmark indices for eyes (from the 68-point model)
LEFT_EYE_IDX = list(range(42, 48))   # dlib indexing: 42-47 (right eye visually, but left in image coords if mirrored)
RIGHT_EYE_IDX = list(range(36, 42))  # 36-41

# ---------- Utilities ----------
def euclidean(a, b):
    a = np.array(a); b = np.array(b)
    return np.linalg.norm(a - b)

def play_tone(frequency=440.0, duration_s=0.4, volume=0.2):
    """Play simple tone if simpleaudio is available."""
    if not AUDIO_AVAILABLE:
        print("[Audio] simpleaudio not available. Skipping tone.")
        return None
    fs = 44100
    t = np.linspace(0, duration_s, int(fs * duration_s), False)
    wave = np.sin(2 * np.pi * frequency * t)
    audio = (wave * (2**15 - 1) * volume).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    return play_obj

# ---------- Alert Manager ----------
class AlertManager:
    def __init__(self, mqtt_broker=MQTT_BROKER, mqtt_port=MQTT_PORT):
        self.state = "green"
        self.alarm_running = False
        self.escalation_thread = None
        self.mqtt_client = None
        if MQTT_AVAILABLE:
            try:
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(mqtt_broker, mqtt_port, 60)
                self.mqtt_client.loop_start()
            except Exception as e:
                print(f"[AlertManager] MQTT connect failed: {e}")
                self.mqtt_client = None

    def set_light(self, color: str):
        self.state = color
        payload = {"color": color}
        if self.mqtt_client:
            try:
                self.mqtt_client.publish(MQTT_TOPIC_LIGHTS, json.dumps(payload))
            except Exception:
                pass
        print(f"[Alert] Light -> {color}")

    def vibrate(self, intensity=50, duration_ms=300):
        payload = {"intensity": int(intensity), "duration_ms": int(duration_ms)}
        if self.mqtt_client:
            try:
                self.mqtt_client.publish(MQTT_TOPIC_VIBRATION, json.dumps(payload))
            except Exception:
                pass
        print(f"[Alert] Vibrate -> {payload}")

    def start_escalating_alarm(self):
        if self.alarm_running:
            return
        self.alarm_running = True
        def escalate():
            print("[Alarm] escalation started")
            level = 0
            while self.alarm_running and level < 12:
                freq = 700 + level * 100
                vol = min(0.06 + level * 0.08, 1.0)
                play_tone(frequency=freq, duration_s=0.6, volume=vol)
                self.vibrate(intensity=30 + level*6, duration_ms=200 + level*40)
                time.sleep(0.75)
                level += 1
            # if still running, sustain loud beeps
            while self.alarm_running:
                play_tone(frequency=1400, duration_s=0.4, volume=1.0)
                self.vibrate(intensity=100, duration_ms=500)
                time.sleep(0.5)
            print("[Alarm] escalation stopped")
        self.escalation_thread = threading.Thread(target=escalate, daemon=True)
        self.escalation_thread.start()

    def stop_alarm(self):
        self.alarm_running = False
        self.set_light("green")
        print("[AlertManager] Alarm stopped")
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass

# ---------- Drowsiness & Lazy-eye Detector ----------
class DlibDriverMonitor:
    def __init__(self, predictor_path):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Landmark predictor not found: {predictor_path}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.drowsy_counter = 0

    @staticmethod
    def eye_aspect_ratio(eye):
        # eye is Nx2 array of (x,y) points. For 6 points: p0..p5 like the standard formula
        A = euclidean(eye[1], eye[5])
        B = euclidean(eye[2], eye[4])
        C = euclidean(eye[0], eye[3])
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return ear

    def detect(self, gray_frame):
        """
        Input: gray_frame (numpy array)
        Returns: dict with keys:
            - drowsy (bool)
            - left_ear, right_ear (floats)
            - strabismus_flag (bool)
            - landmarks (list of (x,y))
            - rect (dlib rect of face) or None
        """
        res = {"drowsy": False, "left_ear": None, "right_ear": None,
               "strabismus_flag": False, "landmarks": None, "rect": None}
        rects = self.detector(gray_frame, 0)
        if len(rects) == 0:
            return res
        rect = rects[0]  # first face only
        shape = self.predictor(gray_frame, rect)
        shape_np = face_utils.shape_to_np(shape)
        res["landmarks"] = shape_np
        res["rect"] = rect

        left_eye = shape_np[LEFT_EYE_IDX]
        right_eye = shape_np[RIGHT_EYE_IDX]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        res["left_ear"] = left_ear
        res["right_ear"] = right_ear

        ear_mean = (left_ear + right_ear) / 2.0
        if ear_mean < EAR_THRESHOLD:
            self.drowsy_counter += 1
        else:
            self.drowsy_counter = 0
        res["drowsy"] = (self.drowsy_counter >= CONSEC_FRAMES)

        # Simple lazy-eye screening heuristic:
        left_center = left_eye.mean(axis=0)
        right_center = right_eye.mean(axis=0)
        interocular = euclidean(left_center, right_center)
        face_mid_x = (left_center[0] + right_center[0]) / 2.0
        left_offset = abs(left_center[0] - face_mid_x) / interocular if interocular > 0 else 0.0
        right_offset = abs(right_center[0] - face_mid_x) / interocular if interocular > 0 else 0.0
        asym = abs(left_offset - right_offset)
        res["strabismus_flag"] = (asym > STRABISMUS_ASYM_THRESHOLD)
        return res

# ---------- Main app ----------
def main(args):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Optional video file to use instead of webcam")
    parser.add_argument("--predictor", default=SHAPE_PREDICTOR_PATH, help="Path to dlib shape predictor .dat")
    parsed = parser.parse_args(args)

    # initialize detector & alerts
    monitor = DlibDriverMonitor(parsed.predictor)
    alerts = AlertManager()

    # video capture: prefer webcam (if OpenCV installed), else video file
    cap = None
    if parsed.video:
        if OPENCV_AVAILABLE:
            cap = cv2.VideoCapture(parsed.video)
        else:
            raise RuntimeError("Video file specified but OpenCV not available for playback.")
    else:
        if OPENCV_AVAILABLE:
            cap = cv2.VideoCapture(0)
            time.sleep(1.0)
        else:
            raise RuntimeError("No webcam available without OpenCV. Install opencv-python or specify a video file with --video.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] No frame received. Exiting.")
                break
            # For consistent detection use grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detection = monitor.detect(gray)
            # HUD messages
            hud = []
            if detection["left_ear"] is not None:
                hud.append(f"EAR L:{detection['left_ear']:.3f} R:{detection['right_ear']:.3f}")

            # Drowsiness handling
            if detection["drowsy"]:
                hud.append("DROWSY!")
                if alerts.state != "red":
                    alerts.set_light("red")
                    # first light/haptic
                    alerts.vibrate(intensity=70, duration_ms=400)
                    # start escalating audible alarm
                    alerts.start_escalating_alarm()
                # Acknowledge if driver opens eyes wide
                if (detection["left_ear"] + detection["right_ear"]) / 2.0 > (EAR_THRESHOLD + 0.08):
                    print("[Main] Eyes opened -> acknowledging")
                    alerts.stop_alarm()
            else:
                # step down colors gently
                if alerts.state == "red":
                    alerts.set_light("yellow")
                    alerts.vibrate(intensity=30, duration_ms=250)
                elif alerts.state == "yellow":
                    alerts.set_light("green")
                # lazy-eye screening (non-alarming)
                if detection["strabismus_flag"]:
                    hud.append("EYE ASYMMETRY: SCREEN")
                    # gentle notice
                    alerts.set_light("yellow")
                    alerts.vibrate(intensity=35, duration_ms=200)

            # Draw landmarks & HUD
            if detection["landmarks"] is not None:
                for (x, y) in detection["landmarks"]:
                    cv2.circle(frame, (int(x), int(y)), 1, (0,255,0), -1)
            for i, msg in enumerate(hud):
                cv2.putText(frame, msg, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Driver Monitor (dlib)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        if cap:
            cap.release()
        if OPENCV_AVAILABLE:
            cv2.destroyAllWindows()
        alerts.stop_alarm()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
