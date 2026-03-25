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

# Tải cấu hình
load_dotenv()

# Các module Custom (Đảm bảo các file này tồn tại trong cùng thư mục)
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Calibration import Calibrator
from PredictMaker import DecisionMaker

# --- KHỞI TẠO MÔ HÌNH ---
print("[INFO] Đang nạp các mô hình AI và mô hình nhận diện (ResNet)...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./dlib_shape_predictor/dlib_face_recognition_resnet_model_v1.dat')

# Cấu hình AI
calibrator = Calibrator(required_frames=100) 
decision_maker = DecisionMaker(window_size=30, model_path="Models/dms_model_int8.tflite")

# --- KẾT NỐI SUPABASE ---
user_id = os.getenv("USER_ID", "00000000-0000-0000-0000-000000000000")
trip_id = os.getenv("TRIP_ID", "11111111-1111-1111-1111-111111111111")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except:
    supabase_client = None
    print("[WARNING] Supabase chưa được cấu hình. Chạy ở chế độ Offline.")

# --- ĐỊNH NGHĨA BIẾN HỖ TRỢ ---
frame_width, frame_height = 480, 480
image_points = np.zeros((6, 2), dtype="double")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mStart, mEnd = 49, 68

# Tracking Biến
frame_count = 0
last_calibration_time = time.time()
calibration_interval = 60
no_face_start_time = None
NO_FACE_TIMEOUT = 2.0

# Trip Analytics
total_yawn_count = 0 
total_head_tilt_count = 0
total_eye_closed_time = 0.0
last_sync_time = time.time()

# Buffer & Folders
os.makedirs("temp_alert/images", exist_ok=True)
os.makedirs("temp_alert/videos", exist_ok=True)
frame_buffer = deque(maxlen=60)

# Trạng thái Event hiện tại
current_event_type = "Normal"
event_start_time = None
alert_sent = False

# --- HÀM THỰC THI SUPABASE (THREADED) ---
def update_trip_analytics(t_id, yawns, head_tilts, eye_time):
    if not supabase_client: return
    try:
        level = "Safe"
        if eye_time > 15.0 or yawns > 10: level = "Khẩn cấp"
        elif eye_time > 5.0 or yawns > 3 or head_tilts > 5: level = "Nguy hiểm"
        
        payload = {"total_yawn_count": yawns, "total_eye_closed_time": float(eye_time), "total_head_tilt": head_tilts, "fatigue_level": level}
        supabase_client.table("trips").update(payload).eq("id", t_id).execute()
        print(f"-> [SUPABASE] Đồng bộ trips thành công.")
    except Exception as e: print(f"[ERROR] Sync Trip: {e}")

def process_and_upload_alert(t_id, u_id, e_type, sev, dur, file_img, frames, file_vid):
    # 1. Xuất Video Local
    if frames:
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(file_vid, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))
        for f in frames: out.write(f)
        out.release()

    if not supabase_client: return
    try:
        # 2. Upload Ảnh & Video
        supabase_client.storage.from_("alerts_media").upload(file_img, file_img)
        supabase_client.storage.from_("alerts_media").upload(file_vid, file_vid)
        img_url = supabase_client.storage.from_("alerts_media").get_public_url(file_img)
        vid_url = supabase_client.storage.from_("alerts_media").get_public_url(file_vid)

        # 3. Insert Database (Bọc kỹ để tránh crash do RLS)
        row = {"trip_id": t_id, "user_id": u_id, "alert_type": e_type, "severity": sev, "duration_seconds": float(dur), "image_url": img_url, "video_url": vid_url, "created_at": datetime.now().isoformat()}
        try:
            supabase_client.table("alerts").insert(row).execute()
            print(f"[SUPABASE] Gửi cảnh báo {e_type} thành công.")
        except Exception as e_sql: print(f"[ERROR] Insert SQL: {e_sql}")
    except Exception as e_api: print(f"[ERROR] Supabase API: {e_api}")

# --- VÒNG LẶP CHÍNH ---
print("[INFO] Khởi động DMS...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

try:
    while True:
        frame = vs.read()
        if frame is None: break
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        # UI Cơ bản: Đồng hồ thời gian thực (Góc trên bên phải)
        current_dt = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        cv2.putText(frame, current_dt, (330, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # --- QUẢN LÝ ĐỊNH DANH (IDENTIFICATION) ---
        driver_rect = None
        if not calibrator.is_calibrated:
            if len(rects) == 1: driver_rect = rects[0]
            else: cv2.putText(frame, "WAITING FOR DRIVER...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        else:
            if len(rects) == 1: driver_rect = rects[0]
            elif len(rects) > 1:
                # Chỉ check Face ID khi cần thiết để mượt CPU
                if frame_count % 30 == 0:
                    for r in rects:
                        shape_tmp = predictor(gray, r)
                        enc = np.array(face_encoder.compute_face_descriptor(frame, shape_tmp))
                        if calibrator.is_driver(enc): driver_rect = r; break
                else: driver_rect = rects[0]

        # --- XỬ LÝ MẤT KHUÔN MẶT (DISTRACTED) ---
        if driver_rect is None:
            if calibrator.is_calibrated:
                if no_face_start_time is None: no_face_start_time = time.time()
                else:
                    dur = time.time() - no_face_start_time
                    state, color = "Distracted", (0,0,255)
                    cv2.putText(frame, f"NO FACE: {dur:.1f}s", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Chụp ảnh khi mất mặt quá lâu (Ví dụ > 3s)
                    if dur > 3.0 and not alert_sent:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_p, vid_p = f"temp_alert/images/NF_{ts}.jpg", f"temp_alert/videos/NF_{ts}.mp4"
                        cv2.imwrite(img_p, frame)
                        threading.Thread(target=process_and_upload_alert, args=(trip_id, user_id, "head_tilt", "warning", dur, img_p, list(frame_buffer), vid_p)).start()
                        alert_sent = True
                        current_event_type = "Distracted" # Giữ trạng thái để không log lặp lại
            
            # Hiển thị và tiếp tục
            frame_buffer.append(frame.copy())
            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            continue

        no_face_start_time = None # Reset
        
        # --- TRÍCH XUẤT ĐẶC TRƯNG ---
        shape_obj = predictor(gray, driver_rect)
        shape = face_utils.shape_to_np(shape_obj)
        
        # EAR & MAR
        ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0
        mar = mouth_aspect_ratio(shape[mStart:mEnd])
        
        # Head Pose (Image Points)
        for idx, lm in enumerate([33, 8, 36, 45, 48, 54]):
            image_points[idx] = shape[lm]
        (h_deg, y_deg, p_raw, start_p, end_p, end_p2) = getHeadTiltAndCoords(gray.shape, image_points, frame_height)
        pitch = h_deg[0] if h_deg else 0.0

        # --- AI PIPELINE & CALIBRATION ---
        display_state, display_color = "Normal", (0,255,0)
        
        if not calibrator.is_calibrated:
            calibrator.update(ear, mar, p_raw)
            calibrator.update_face(np.array(face_encoder.compute_face_descriptor(frame, shape_obj)))
            cv2.putText(frame, f"LEARNING: {calibrator.get_progress()*100:.1f}%", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            decision_maker.update_buffer(ear, mar, pitch, y_deg, p_raw, calibrator.ear_baseline, calibrator.mar_baseline, calibrator.pitch_raw_baseline)
            display_state = decision_maker.predict_state()
            if display_state in ["Drowsy", "Distracted"]: display_color = (0,0,255)
            elif display_state == "Yawning": display_color = (0,165,255)

        # UI Overlay
        cv2.putText(frame, f"STATE: {display_state}", (140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        cv2.putText(frame, f"Y: {total_yawn_count} ", (430, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"D: {total_head_tilt_count}", (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"E: {total_eye_closed_time:.1f}s", (430, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"P: {pitch:.1f}deg", (430, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # --- QUẢN LÝ SỰ KIỆN CẢNH BÁO ---
        state = display_state # Ghi đè để logic mượt dùng kết quả AI
        if state != "Normal" and state != "Talking":
            if current_event_type != state:
                current_event_type, event_start_time, alert_sent = state, time.time(), False
                print(f"\n[!] BẮT ĐẦU: {state}")
            
            dur_sec = time.time() - event_start_time
            cv2.putText(frame, f"+{dur_sec:.1f}s", (140, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)
            
            # Gửi cảnh báo nếu vượt ngưỡng (Drowsy > 1.5s, Distracted/Yawning > 3s)
            if not alert_sent:
                limit = 1.5 if state == "Drowsy" else 3.0
                if dur_sec > limit:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_path, vid_path = f"temp_alert/images/A_{ts}.jpg", f"temp_alert/videos/V_{ts}.mp4"
                    cv2.imwrite(img_path, frame)
                    threading.Thread(target=process_and_upload_alert, args=(trip_id, user_id, state.lower(), "danger", dur_sec, img_path, list(frame_buffer), vid_path)).start()
                    alert_sent = True
        else:
            if current_event_type != "Normal":
                dur = time.time() - event_start_time
                print(f"[#] KẾT THÚC: {current_event_type} ({dur:.1f}s)")
                if current_event_type == "Drowsy": total_eye_closed_time += dur
                elif current_event_type == "Yawning": total_yawn_count += 1
                elif current_event_type == "Distracted": total_head_tilt_count += 1
                current_event_type, event_start_time, alert_sent = "Normal", None, False

        # --- SYNC DATA & LOOP FB ---
        if (time.time() - last_sync_time) > 60:
            threading.Thread(target=update_trip_analytics, args=(trip_id, total_yawn_count, total_head_tilt_count, total_eye_closed_time)).start()
            last_sync_time = time.time()
            if calibrator.is_calibrated and (time.time() - last_calibration_time > calibration_interval):
                calibrator.reset(); last_calibration_time = time.time(); print("[RE-CALIBRATE]")

        frame_buffer.append(frame.copy())
        cv2.imshow("DMS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    print("[INFO] Đang đóng hệ thống...")
    cv2.destroyAllWindows()
    vs.stop()