#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2  
import numpy as np
import threading
from datetime import datetime
import os
from dotenv import load_dotenv  
from collections import deque
from supabase import create_client, Client

# [CẬP NHẬT 1]: Thêm thư viện Flask để phát luồng Video
from flask import Flask, Response

load_dotenv()

# Các module Custom
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Calibration import Calibrator
from PredictMaker import DecisionMaker

# [CẬP NHẬT 2]: Khởi tạo Server Flask
app = Flask(__name__)

print("[INFO] Đang nạp bộ dự đoán landmark trên khuôn mặt...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

print("[INFO] Đang khởi tạo hệ thống AI và Calibrator...")
calibrator = Calibrator(required_frames=100) 
decision_maker = DecisionMaker(window_size=30, model_path="Models/dms_model_int8.tflite")

print("[INFO] Khởi động Camera...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

image_points = np.array([
    (359, 391),     
    (399, 561),     
    (337, 297),     
    (513, 301),     
    (345, 465),     
    (453, 469)      
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

last_calibration_time = time.time()
calibration_interval = 60 
previous_state = "Normal"

user_id = os.getenv("USER_ID", "00000000-0000-0000-0000-000000000000")  
trip_id = os.getenv("TRIP_ID", "11111111-1111-1111-1111-111111111111")  
event_start_time = None
current_event_type = "Normal"
total_yawn_count = 0  
total_head_tilt_count = 0
total_eye_closed_time = 0.0 
alert_sent = False  
last_sync_time = time.time() 

os.makedirs("temp_alert/images", exist_ok=True)
os.makedirs("temp_alert/videos", exist_ok=True)
frame_buffer = deque(maxlen=60)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")  
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[CẢNH BÁO] Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY trong file .env!")
    supabase_client = None
else:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def update_trip_analytics(trip_id, yawns, head_tilts, eye_closed_time):
    if supabase_client is None: return
    try:
        fatigue_level = "Safe"
        if eye_closed_time > 15.0 or yawns > 10: fatigue_level = "Khẩn cấp"
        elif eye_closed_time > 5.0 or yawns > 3 or head_tilts > 5: fatigue_level = "Nguy hiểm"
        
        payload = {
            "total_yawn_count": yawns,
            "total_eye_closed_time": float(eye_closed_time),
            "total_head_tilt": head_tilts,
            "fatigue_level": fatigue_level
        }
        supabase_client.table("trips").update(payload).eq("id", trip_id).execute()
        print(f"-> [SUPABASE-CRON] Đã Update thành công bảng TRIPS!")
    except Exception as e:
        print(f"\n[SUPABASE-ERROR] Lỗi khi cập nhật bảng trips: {e}")

def process_and_upload_alert(trip_id, user_id, event_type, severity, duration, filename_img, frames_list, filename_vid):
    if frames_list:
        height, width, layers = frames_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename_vid, fourcc, 15.0, (width, height))
        for f in frames_list: out.write(f)
        out.release()
        print(f"\n[LOCAL] Đã xuất video vòng lặp: {filename_vid}")

    if supabase_client is None:
        print("\n[CẢNH BÁO] Chưa điền API Key Supabase! Hệ thống kết thúc offline.")
        return

    try:
        print(f"\n[SUPABASE-SYNC] Đang tải bằng chứng lên đám mây Storage (alerts_media)...")
        supabase_client.storage.from_("alerts_media").upload(filename_img, filename_img)
        img_url = supabase_client.storage.from_("alerts_media").get_public_url(filename_img)

        supabase_client.storage.from_("alerts_media").upload(filename_vid, filename_vid)
        vid_url = supabase_client.storage.from_("alerts_media").get_public_url(filename_vid)

        data_payload = {
            "trip_id": trip_id,
            "user_id": user_id,
            "alert_type": event_type,
            "severity": severity,
            "duration_seconds": float(duration),
            "image_url": img_url,
            "video_url": vid_url, 
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase_client.table("alerts").insert(data_payload).execute()
        print(f"\n[SUPABASE-SUCCESS] Đã Insert Alert & Gửi Notification cho App thành công!")
    except Exception as e:
        print(f"\n[SUPABASE-ERROR] Lỗi API: {e}")


# [CẬP NHẬT 3]: Bọc toàn bộ vòng lặp while vào một hàm Generator của Flask
def phat_video():
    # Khai báo global để vòng lặp có quyền chỉnh sửa các biến đếm bên ngoài
    global last_calibration_time, current_event_type, event_start_time, alert_sent
    global total_yawn_count, total_head_tilt_count, total_eye_closed_time
    global last_sync_time, previous_state, image_points

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1024, height=576)

        current_time = time.time()
        if calibrator.is_calibrated and (current_time - last_calibration_time > calibration_interval):
            print(f"\n[HỆ THỐNG] Tự động kích hoạt lại chu kỳ học sinh trắc học sau mỗi {calibration_interval} giây!")
            calibrator.reset()
            last_calibration_time = current_time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        rects = detector(gray, 0)

        if len(rects) > 0:
            cv2.putText(frame, f"{len(rects)} face(s) found", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for rect in rects:
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            for (i, (x, y)) in enumerate(shape):
                if i == 33: image_points[0] = np.array([x, y], dtype='double')
                elif i == 8: image_points[1] = np.array([x, y], dtype='double')
                elif i == 36: image_points[2] = np.array([x, y], dtype='double')
                elif i == 45: image_points[3] = np.array([x, y], dtype='double')
                elif i == 48: image_points[4] = np.array([x, y], dtype='double')
                elif i == 54: image_points[5] = np.array([x, y], dtype='double')

            (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
            pitch = head_tilt_degree[0] if head_tilt_degree else 0.0

            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

            if not calibrator.is_calibrated:
                calibrator.update(ear, mar)
                progress = calibrator.get_progress() * 100
                cv2.putText(frame, f"CALIBRATING AI... {progress:.1f}%", (320, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            else:
                decision_maker.update_buffer(
                    ear, mar, pitch, 
                    ear_baseline=calibrator.ear_baseline, 
                    mar_baseline=calibrator.mar_baseline
                )
                
                state = decision_maker.predict_state()
                
                cv2.putText(frame, f"EAR diff: {ear - calibrator.ear_baseline:.3f}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.putText(frame, f"MAR diff: {mar - calibrator.mar_baseline:.3f}", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                color = (0, 255, 0) 
                if state in ["Drowsy", "Distracted"]: color = (0, 0, 255) 
                elif state == "Yawning": color = (0, 165, 255) 
                elif state == "Talking": color = (255, 255, 0) 
                
                cv2.putText(frame, f"AI STATE: {state}", (350, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                overlay = frame.copy()
                cv2.rectangle(overlay, (700, 10), (1010, 150), (0, 0, 0), -1)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.putText(frame, f"TRIP ANALYTICS", (720, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yawns: {total_yawn_count}", (720, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Distracted: {total_head_tilt_count}", (720, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(frame, f"Eyes Closed: {total_eye_closed_time:.1f}s", (720, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if current_event_type != "Normal" and event_start_time is not None:
                    current_duration = time.time() - event_start_time
                    alert_text = f"+ {current_duration:.1f}s"
                    color_alert = (0, 0, 255) if current_duration > 2.0 else (0, 165, 255)
                    cv2.putText(frame, alert_text, (350, 95), cv2.FONT_HERSHEY_DUPLEX, 1.2, color_alert, 2)

                current_dt = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                cv2.putText(frame, current_dt, (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                if state != "Normal" and state != "Talking":
                    if current_event_type != state: 
                        current_event_type = state
                        event_start_time = time.time()
                        alert_sent = False
                        
                        if state == "Drowsy": print("\n[NGUY HIỂM] Có dấu hiệu nhắm mắt (Drowsy)!")
                        elif state == "Distracted": print("\n[CẢNH BÁO] Mất tập trung (Distracted)!")
                        elif state == "Yawning": print("\n[NHẮC NHỞ] Đang ngáp (Yawning)!")

                    if event_start_time is not None:
                        duration_seconds = time.time() - event_start_time
                        
                        if current_event_type == "Drowsy" and duration_seconds > 2.0 and not alert_sent:
                            severity = "emergency" if duration_seconds > 4.0 else "danger"
                            
                            filename = f"temp_alert/images/alert_drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            vid_filename = f"temp_alert/videos/alert_drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            cv2.imwrite(filename, frame)
                            print(f">>> [LƯU ẢNH CAMERA] Lưu file {filename} thành công. Đã ngủ: {duration_seconds:.1f}s")
                            
                            frames_to_save = list(frame_buffer)
                            threading.Thread(target=process_and_upload_alert, args=(
                                trip_id, user_id, "eyes_closed", severity, duration_seconds, filename, frames_to_save, vid_filename
                            )).start()
                            alert_sent = True 
                            
                        elif current_event_type == "Distracted" and duration_seconds > 4.0 and not alert_sent:
                            filename = f"temp_alert/images/alert_distracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            vid_filename = f"temp_alert/videos/alert_distracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            cv2.imwrite(filename, frame)
                            print(f">>> [LƯU ẢNH CAMERA] Lưu file {filename} do mất tập trung.")
                            
                            frames_to_save = list(frame_buffer)
                            threading.Thread(target=process_and_upload_alert, args=(
                                trip_id, user_id, "head_tilt", "warning", duration_seconds, filename, frames_to_save, vid_filename
                            )).start()
                            alert_sent = True
                        
                else: 
                    if current_event_type != "Normal":
                        duration = time.time() - event_start_time if event_start_time else 0
                        print(f"\n[THÔNG TIN] Tài xế đã tỉnh / Tập trung lại. Tổng t.gian sự kiện vừa rồi: {duration:.1f}s")
                        
                        if current_event_type == "Drowsy":
                            total_eye_closed_time += duration
                        elif current_event_type == "Yawning":
                            total_yawn_count += 1
                        elif current_event_type == "Distracted":
                            total_head_tilt_count += 1
                            
                        current_event_type = "Normal"
                        event_start_time = None
                        alert_sent = False
                        
                if (time.time() - last_sync_time) > 60: 
                    print(f"\n=== [SUPABASE-ANALYTICS] GỌI API ĐỒNG BỘ BẢNG TRIPS ({trip_id}) ===")
                    threading.Thread(target=update_trip_analytics, args=(trip_id, total_yawn_count, total_head_tilt_count, total_eye_closed_time)).start()
                    last_sync_time = time.time()
                    
                previous_state = state

        frame_buffer.append(frame.copy())

        # [CẬP NHẬT 4]: Xóa lệnh cv2.imshow và thay bằng lệnh đẩy Video lên mạng
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# [CẬP NHẬT 5]: Tạo đường dẫn (URL) cho App Android kết nối vào
@app.route('/video_feed')
def video_feed():
    return Response(phat_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n[HỆ THỐNG] MÁY CHỦ VIDEO ĐANG CHẠY...")
    print("[HỆ THỐNG] Hãy dùng Ngrok để chuyển tiếp cổng 5000 (Lệnh: ngrok http 5000)")
    # Chạy Server trên đa luồng (threaded=True) để không bị nghẽn
    app.run(host='0.0.0.0', port=5000, threaded=True)