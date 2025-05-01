from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import deque
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)

# === Load trained model and scaler ===
model = load_model("neural_posture_model.h5")
scaler = joblib.load("scaler.pkl")

# === Expected features (must match training data) ===
expected_features = [
    "view_front", "view_left", "view_right",
    "NOSE_x", "NOSE_y",
    "LEFT_EAR_x", "LEFT_EAR_y",
    "RIGHT_EAR_x", "RIGHT_EAR_y",
    "LEFT_SHOULDER_x", "LEFT_SHOULDER_y",
    "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y"
]

# === MediaPipe setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Webcam setup ===
cap = cv2.VideoCapture(0)
prediction_buffer = deque(maxlen=10)
posture_log = []
current_posture = "Detecting..."
posture_stats = {}
session_start_time = time.time()

def gen_frames():
    global current_posture, posture_stats
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                lm = results.pose_landmarks.landmark
                view = {"view_front": 1, "view_left": 0, "view_right": 0}

                def get_xy(part):
                    landmark = lm[getattr(mp_pose.PoseLandmark, part)]
                    if landmark.visibility < 0.5:
                        raise ValueError(f"Low visibility: {part}")
                    return [landmark.x, landmark.y]

                data = {
                    **view,
                    "NOSE_x": get_xy("NOSE")[0],
                    "NOSE_y": get_xy("NOSE")[1],
                    "LEFT_EAR_x": get_xy("LEFT_EAR")[0],
                    "LEFT_EAR_y": get_xy("LEFT_EAR")[1],
                    "RIGHT_EAR_x": get_xy("RIGHT_EAR")[0],
                    "RIGHT_EAR_y": get_xy("RIGHT_EAR")[1],
                    "LEFT_SHOULDER_x": get_xy("LEFT_SHOULDER")[0],
                    "LEFT_SHOULDER_y": get_xy("LEFT_SHOULDER")[1],
                    "RIGHT_SHOULDER_x": get_xy("RIGHT_SHOULDER")[0],
                    "RIGHT_SHOULDER_y": get_xy("RIGHT_SHOULDER")[1],
                }

                input_df = pd.DataFrame([[data[f] for f in expected_features]], columns=expected_features)
                input_scaled = scaler.transform(input_df)
                pred_prob = model.predict(input_scaled)[0][0]
                prediction = "normal" if pred_prob > 0.5 else "forward Head"

                prediction_buffer.append(prediction)
                if prediction_buffer.count(prediction) > 6:
                    current_posture = prediction
                    posture_stats[current_posture] = posture_stats.get(current_posture, 0) + 1
                    now = datetime.now().strftime("%H:%M:%S")
                    if not posture_log or posture_log[-1]['posture'] != current_posture:
                        posture_log.append({'time': now, 'posture': current_posture})

            except Exception as e:
                print(f"[!] Error: {e}")
                current_posture = "Detecting..."

        color = (0, 255, 0) if current_posture == "normal" else (0, 0, 255)
        cv2.putText(frame, f"Posture: {current_posture}", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    user_profile = {"name": "Phyo Theingi"}
    session_duration = int((time.time() - session_start_time) / 60)
    total_frames = sum(posture_stats.values())
    posture_percentages = {
        posture: round((count / total_frames) * 100, 1) if total_frames else 0
        for posture, count in posture_stats.items()
    }

    return render_template(
        'dashboard.html',
        user=user_profile,
        log=posture_log,
        current_posture=current_posture,
        posture_stats=posture_percentages,
        session_duration=session_duration,
        posture_types=list(posture_stats.keys())
    )

@app.route('/stats')
def stats():
    total_frames = sum(posture_stats.values())
    posture_percentages = {
        posture: round((count / total_frames) * 100, 1) if total_frames else 0
        for posture, count in posture_stats.items()
    }
    return jsonify({
        'current_posture': current_posture,
        'posture_stats': posture_percentages,
        'session_duration': int((time.time() - session_start_time) / 60),
        'posture_types': list(posture_stats.keys())
    })

@app.route('/reset')
def reset_session():
    global posture_log, posture_stats, session_start_time
    posture_log = []
    posture_stats = {}
    session_start_time = time.time()
    return jsonify({'status': 'success'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)