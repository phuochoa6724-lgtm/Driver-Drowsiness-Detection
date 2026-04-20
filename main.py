from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import threading
from collections import deque
import os
from dotenv import load_dotenv
from flask import Flask, Response, jsonify

load_dotenv()

# --- IMPORT CÁC MODULE TÙY CHỈNH ---
from src.detection.ear import eye_aspect_ratio
from src.detection.mar import mouth_aspect_ratio
from src.detection.head_pose import getHeadTiltAndCoords
from src.core.calibration import Calibrator
from src.inference.predict_maker import DecisionMaker
from src.core.backend import BackendManager
from src.core.alert_handler import AlertHandler
from src.ui.ui_helper import UIHelper

# ============================================================
# FLASK SERVER
# ============================================================
app = Flask(__name__)

_state_lock = threading.Lock()
_shared_state = {
    "current_state": "Normal",
    "yawns": 0,
    "eyes_closed_time": 0.0,
    "head_tilts": 0,
}
_latest_frame = None
_frame_lock = threading.Lock()

@app.route("/api/data")
def api_data():
    with _state_lock:
        return jsonify(_shared_state)

@app.route("/snapshot")
def snapshot():
    with _frame_lock:
        frame = _latest_frame
    if frame is None:
        return "", 204
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def _run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

threading.Thread(target=_run_flask, daemon=True).start()
print("[INFO] Flask server đang chạy tại port 5000")
# ============================================================

# --- KHỞI TẠO CÁC THÀNH PHẦN HỆ THỐNG ---
backend = BackendManager()
alert_handler = AlertHandler(backend)
ui = UIHelper()

# --- KHỞI TẠO MÔ HÌNH AI & PHẦN CỨNG ---
print("[INFO] Đang nạp các mô hình AI và nhận diện khuôn mặt...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('models/dlib/dlib_face_recognition_resnet_model_v1.dat')

calibrator = Calibrator(required_frames=100)
decision_maker = DecisionMaker(window_size=15, model_path="models/tflite/dms_model_int8.tflite")

frame_width, frame_height = 320, 320
image_points = np.zeros((6, 2), dtype="double")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mStart, mEnd = 49, 68

frame_count = 0
last_sync_time = time.time()
calibration_voice_played = False
frame_buffer = deque(maxlen=60)

print("[INFO] Khởi động hệ thống Driver Monitoring System (DMS)...")

camera_src_env = os.getenv("CAMERA_SOURCE", "0")
try:
    camera_src = int(camera_src_env)
except ValueError:
    camera_src = camera_src_env

vs = VideoStream(src=camera_src).start()
time.sleep(2.0)

last_driver_rect = None
cached_rects = []

try:
    while True:
        frame = vs.read()
        if frame is None: break
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 3 == 1 or len(cached_rects) == 0:
            rects = detector(gray, 0)
            cached_rects = rects
        else:
            rects = cached_rects

        ui.draw_clock(frame)

        driver_rect = None
        if not calibrator.is_calibrated:
            if len(rects) == 1: driver_rect = rects[0]
            else: ui.draw_warning_text(frame, "WAITING FOR DRIVER...")
        else:
            if len(rects) == 1:
                driver_rect = rects[0]
            elif len(rects) > 1 and frame_count % 30 == 0:
                for r in rects:
                    shape_tmp = predictor(gray, r)
                    enc = np.array(face_encoder.compute_face_descriptor(frame, shape_tmp))
                    if calibrator.is_driver(enc):
                        driver_rect = r
                        break
            elif len(rects) > 1:
                if last_driver_rect is not None:
                    last_center = ((last_driver_rect.left() + last_driver_rect.right()) / 2,
                                   (last_driver_rect.top() + last_driver_rect.bottom()) / 2)
                    min_dist = float('inf')
                    for r in rects:
                        center = ((r.left() + r.right()) / 2, (r.top() + r.bottom()) / 2)
                        dist = (center[0] - last_center[0])**2 + (center[1] - last_center[1])**2
                        if dist < min_dist:
                            min_dist = dist
                            driver_rect = r
                else:
                    driver_rect = rects[0]

        if driver_rect is not None:
            last_driver_rect = driver_rect

        if driver_rect is None:
            if calibrator.is_calibrated:
                alert_handler.process_state("Distracted", frame, frame_buffer)
                if alert_handler.current_event == "Distracted":
                    ui.draw_status(frame, "Distracted", (0, 0, 255))

            with _frame_lock:
                _latest_frame = frame.copy()

            frame_buffer.append(frame.copy())
            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            continue

        shape_obj = predictor(gray, driver_rect)
        shape = face_utils.shape_to_np(shape_obj)

        ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0
        mar = mouth_aspect_ratio(shape[mStart:mEnd])

        for idx, lm in enumerate([30, 27, 36, 45, 39, 42]):
            image_points[idx] = shape[lm]
        (h_deg, y_deg, p_raw, start_p, end_p, end_p2) = getHeadTiltAndCoords(gray.shape, image_points, frame_height)
        pitch = h_deg[0] if len(h_deg) > 0 else 0.0

        yaw_rad = abs(y_deg) * np.pi / 180.0
        mar_corrected = mar * float(np.cos(yaw_rad))

        display_state, display_color = "Normal", (0, 255, 0)

        if not calibrator.is_calibrated:
            if not calibration_voice_played:
                alert_handler.play_calibration_reminder()
                calibration_voice_played = True
            calibrator.update(ear, mar_corrected, p_raw)
            calibrator.update_face(np.array(face_encoder.compute_face_descriptor(frame, shape_obj)))
            ui.draw_calibration_progress(frame, calibrator.get_progress())
        else:
            decision_maker.update_buffer(ear, mar_corrected, pitch, y_deg, p_raw,
                                         calibrator.ear_baseline,
                                         calibrator.mar_baseline,
                                         calibrator.pitch_raw_baseline)
            display_state = decision_maker.predict_state()

            if display_state in ["Drowsy", "Distracted"]: display_color = (0, 0, 255)
            elif display_state == "Yawning": display_color = (0, 165, 255)

        alert_handler.process_state(display_state, frame, frame_buffer)

        # Cập nhật shared state cho Flask
        with _state_lock:
            _shared_state["current_state"] = display_state
            _shared_state["yawns"] = alert_handler.total_yawn_count
            _shared_state["eyes_closed_time"] = round(alert_handler.total_eye_closed_time, 2)
            _shared_state["head_tilts"] = alert_handler.total_head_tilt_count

        # Cập nhật frame mới nhất cho snapshot/stream
        with _frame_lock:
            _latest_frame = frame.copy()

        ui.draw_status(frame, display_state, display_color)
        ui.draw_analytics(frame,
                          alert_handler.total_yawn_count,
                          alert_handler.total_head_tilt_count,
                          alert_handler.total_eye_closed_time,
                          pitch)

        if (time.time() - last_sync_time) > 30:
            threading.Thread(target=backend.update_trip_analytics,
                             args=(alert_handler.total_yawn_count,
                                   alert_handler.total_head_tilt_count,
                                   alert_handler.total_eye_closed_time,
                                   alert_handler.total_drowsy_count,
                                   alert_handler.total_distracted_count)).start()
            last_sync_time = time.time()

        frame_buffer.append(frame.copy())
        cv2.imshow("DMS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    print("[INFO] Đang dừng hệ thống và đóng hành trình...")
    backend.close_session(alert_handler.total_yawn_count,
                          alert_handler.total_head_tilt_count,
                          alert_handler.total_eye_closed_time,
                          alert_handler.total_drowsy_count,
                          alert_handler.total_distracted_count)
    cv2.destroyAllWindows()
    vs.stop()
