#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2  # Nhập thư viện xử lý ảnh OpenCV (Computer Vision)
import numpy as np
import threading
from datetime import datetime
import os
from dotenv import load_dotenv  # Thư viện đọc biến môi trường từ file .env

# Tự động nạp các biến môi trường từ file .env vào os.environ
load_dotenv()
from collections import deque  # Nhập cấu trúc hàng đợi 2 đầu để tạo bộ đệm frame
from supabase import create_client, Client

# Các module Custom
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Calibration import Calibrator
from PredictMaker import DecisionMaker


# Khởi tạo bộ phát hiện khuôn mặt của dlib (dựa trên HOG) và sau đó tạo
# bộ dự đoán các điểm mốc trên khuôn mặt (facial landmark predictor)
print("[INFO] Đang nạp bộ dự đoán landmark trên khuôn mặt...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Mô hình nhận diện khuôn mặt (Face Recognition) để xác định đúng tài xế
print("[INFO] Đang nạp bộ nhận diện khuôn mặt tài xế (Face Recognition)...")
face_encoder = dlib.face_recognition_model_v1('./dlib_shape_predictor/dlib_face_recognition_resnet_model_v1.dat')

# Khởi tạo mô hình AI và bộ lấy mẫu
print("[INFO] Đang khởi tạo hệ thống AI và Calibrator...")
calibrator = Calibrator(required_frames=100) # Chỉ lấy 100 khung hình làm mẫu ban đầu
decision_maker = DecisionMaker(window_size=30, model_path="Models/dms_model_int8.tflite")

# Khởi tạo luồng video (video stream)
print("[INFO] Khởi động Camera...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame_width = 480
frame_height = 480

# Tọa độ mặc định cho việc vẽ line Đầu (Pitch, Yaw, Roll)
image_points = np.array([
    (359, 391),     # Đỉnh mũi 34
    (399, 561),     # Cằm 9
    (337, 297),     # Góc trái của mắt trái 37
    (513, 301),     # Góc phải của mắt phải 46
    (345, 465),     # Góc trái miệng 49
    (453, 469)      # Góc phải miệng 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

# Hỗ trợ tự động căn chỉnh lại chu kỳ học theo thời gian
last_calibration_time = time.time()
calibration_interval = 60 # Khoảng cách lặp vòng học: 300 giây (5 phút)
previous_state = "Normal"

# [1] KHỞI TẠO CÁC BIẾN COUNTER TRƯỚC VÒNG LẶP (Supabase Data)
# Đọc UUID từ file .env thay vì hardcode trực tiếp vào code
user_id = os.getenv("USER_ID", "00000000-0000-0000-0000-000000000000")  # UUID mặc định nếu không tìm thấy trong .env
trip_id = os.getenv("TRIP_ID", "11111111-1111-1111-1111-111111111111")  # UUID mặc định nếu không tìm thấy trong .env
event_start_time = None
current_event_type = "Normal"
total_yawn_count = 0 
total_head_tilt_count = 0
total_eye_closed_time = 0.0 # Bằng giây
alert_sent = False 
last_sync_time = time.time() # Để test gửi luồng 2

# Tạo thư mục lưu ảnh cảnh báo tự động nếu chưa có
os.makedirs("temp_alert/images", exist_ok=True)
os.makedirs("temp_alert/videos", exist_ok=True)

# Bộ đệm để lưu lại 3-4 giây video TRƯỚC VÀ TRONG LÚC vi phạm (khoảng 60 khung hình)
frame_buffer = deque(maxlen=60)

# CẤU HÌNH SUPABASE - Key được đọc từ file .env, KHÔNG hardcode ở đây!
SUPABASE_URL = os.getenv("SUPABASE_URL", "")  # Đọc URL từ biến môi trường
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # Đọc API Key từ biến môi trường

# Chỉ khởi tạo Supabase client nếu cả hai biến đều đã được cấu hình
if not SUPABASE_URL or not SUPABASE_KEY:
    print("[CẢNH BÁO] Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY trong file .env!")
    print("           Hãy sao chép file .env.example thành .env và điền thông tin vào.")
    supabase_client = None
else:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Hàm thread cập nhật thống kê định kỳ bảng trips
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

# Hàm thread Tích hợp: Vừa xuất video MP4, tải lên Storage, ghi vào bảng alerts
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

        # Tùy chọn video - SQL hiện tại bạn chưa khai báo video_url. Nhớ chạy ALTER TABLE public.alerts ADD COLUMN video_url TEXT; 
        supabase_client.storage.from_("alerts_media").upload(filename_vid, filename_vid)
        vid_url = supabase_client.storage.from_("alerts_media").get_public_url(filename_vid)

        # Bắn Data JSON cảnh báo khẩn cấp lên bảng `alerts` chuẩn theo SQL schema
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

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=480, height=480)

    # LUÔN HIỂN THỊ ĐỒNG HỒ THỜI GIAN THỰC (Góc dưới bên trái)
    current_dt = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    cv2.putText(frame, current_dt, (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Biến trạng thái mặc định (để tránh lỗi nếu không tìm thấy mặt)
    state = "Normal"
    color = (0, 255, 0)
    display_state = "Normal"
    display_color = (0, 255, 0)

    # Kiểm tra chu kỳ để tự động lặp lại quy trình học (Re-calibration)
    current_time = time.time()
    if calibrator.is_calibrated and (current_time - last_calibration_time > calibration_interval):
        print(f"\n[HỆ THỐNG] Tự động kích hoạt lại chu kỳ học sinh trắc học sau mỗi {calibration_interval} giây!")
        calibrator.reset()
        last_calibration_time = current_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    # LUÔN HIỂN THỊ SỐ KHUÔN MẶT (Góc trên bên trái)
    if len(rects) > 0:
        cv2.putText(frame, f"{len(rects)} face(s) found", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # ==========================================
    # NHẬN DIỆN VÀ CHỈ XỬ LÝ 1 TÀI XẾ DUY NHẤT
    # ==========================================
    driver_rect = None

    # CHƯA CALIBRATION: Yêu cầu đúng 1 người để lấy mẫu baseline chính xác
    if not calibrator.is_calibrated:
        if len(rects) == 1:
            driver_rect = rects[0]
        elif len(rects) == 0:
            cv2.putText(frame, "CALIBRATING: Khong tim thay tai xe!", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            cv2.putText(frame, f"CALIBRATING: Chi 1 nguoi! ({len(rects)} detected)", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # ĐÃ CALIBRATION XONG: Tìm xem ai là tài xế đã đăng ký
    else:
        if len(rects) == 1:
            # Tối ưu: Nếu chỉ có 1 người, mặc định coi là tài xế để tiết kiệm CPU (không cần tính encoding)
            # Nếu muốn bảo mật hơn có thể tính encoding để check person index
            driver_rect = rects[0]
        elif len(rects) > 1:
            # Có nhiều người (khách hàng, v.v.), cần tìm ra đúng tài xế bằng Face ID
            for rect in rects:
                shape = predictor(gray, rect)
                encoding = np.array(face_encoder.compute_face_descriptor(frame, shape))
                if calibrator.is_driver(encoding):
                    driver_rect = rect
                    break
            
            if driver_rect is None:
                cv2.putText(frame, "KHONG NHAN DIEN DUOC TAI XE", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Nếu không tìm thấy hoặc nhận diện sai tài xế, bỏ qua việc phân tích mệt mỏi/ngáp
    if driver_rect is None:
        # Vẫn đưa vào buffer để stream video không bị giật
        frame_buffer.append(frame.copy())
        try:
            cv2.imshow("Smart AI DMS Architecture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
        except Exception as e:
            print(f"[UI-ERROR] Loi hiển thị: {e}")
        continue

    # BẮT ĐẦU XỬ LÝ CHO TÀI XẾ
    rect = driver_rect
    # Tính toán bounding box
    (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
    
    # Tiên đoán điểm ảnh landmark
    shape_obj = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape_obj)

    # Nếu đang trong quá trình calibration, hãy nạp cả Face Encoding của tài xế vào để lưu mẫu
    if not calibrator.is_calibrated:
        encoding = np.array(face_encoder.compute_face_descriptor(frame, shape_obj))
        calibrator.update_face(encoding)

    # 1. Tính toán Tỷ lệ Mắt (EAR)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    # 2. Tính toán Tỷ lệ Miệng (MAR)
    mouth = shape[mStart:mEnd]
    mar = mouth_aspect_ratio(mouth)
    mouthHull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

    # 3. Tính toán góc đầu (Pitch, Yaw, Roll) bằng hàm bổ trợ
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

    # ==========================================
    # AI PIPELINE CHÍNH THỨC BẮT ĐẦU TẠI ĐÂY
    # ==========================================
    if not calibrator.is_calibrated:
        # Nếu chưa có Baseline cá nhân, đưa vào Module hiệu chuẩn (Calibrator)
        calibrator.update(ear, mar)
        progress = calibrator.get_progress() * 100
        cv2.putText(frame, f"CALIBRATING AI... {progress:.1f}%", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # Hiển thị trạng thái AI và độ chênh lệch (Delta) để dev tiện theo dõi
    # (Mặc định là Normal, sẽ bị ghi đè nếu đã calibrated và có kết quả AI)
    display_state = "Normal"
    display_color = (0, 255, 0)
    
    if calibrator.is_calibrated:
        # Đã có BASELINE. Tính mảng đặc trưng theo Cửa Sổ Trượt (Sliding Window)
        decision_maker.update_buffer(
            ear, mar, pitch, 
            ear_baseline=calibrator.ear_baseline, 
            mar_baseline=calibrator.mar_baseline
        )
        
        # Khởi chạy nội suy AI, lấy nhãn kết quả trạng thái (State)
        display_state = decision_maker.predict_state()
        
        # Cài đặt mã màu hiển thị thông minh
        if display_state in ["Drowsy", "Distracted"]: display_color = (0, 0, 255) # Đỏ (Nguy hiểm)
        elif display_state == "Yawning": display_color = (0, 165, 255) # Cam (Nhắc nhở)
        elif display_state == "Talking": display_color = (255, 255, 0) # Xanh Ngọc (Quan sát)

    # LUÔN HIỂN THỊ CÁC THÔNG SỐ CHÍNH (Kể cả khi đang Calibrating)
    cv2.putText(frame, f"AI STATE: {display_state}", (140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
    
    if calibrator.is_calibrated:
        cv2.putText(frame, f"EAR diff: {ear - calibrator.ear_baseline:.3f}", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        cv2.putText(frame, f"MAR diff: {mar - calibrator.mar_baseline:.3f}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)



    # ==========================================
    # AI PIPELINE CHÍNH THỨC BẮT ĐẦU TẠI ĐÂY (XỬ LÝ LOGIC)
    # ==========================================
    if not calibrator.is_calibrated:
        # Nếu chưa có Baseline cá nhân, đưa vào Module hiệu chuẩn (Calibrator)
        calibrator.update(ear, mar)
        progress = calibrator.get_progress() * 100
        cv2.putText(frame, f"CALIBRATING AI... {progress:.1f}%", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        state = display_state # Gán lại nhãn cho logic cảnh báo bên dưới
        color = display_color


        # ==========================================
        # [VẼ ĐỒ HỌA/OVERLAY LÊN FRAME ĐỂ LƯU KÈM VÀO ẢNH CẢNH BÁO]
        # ==========================================
        # Vẽ bảng nền đen mờ để dễ đọc thông số
        overlay = frame.copy()
        cv2.rectangle(overlay, (300, 40), (475, 140), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # In số liệu thống kê (Analytics) lên góc phải màn hình
        cv2.putText(frame, f"TRIP ANALYTICS", (310, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Yawns: {total_yawn_count}", (310, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, f"Distracted: {total_head_tilt_count}", (310, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.putText(frame, f"Eyes Closed: {total_eye_closed_time:.1f}s", (310, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # In thời gian sự kiện nguy hiểm đang diễn ra (nếu có)
        if current_event_type != "Normal" and event_start_time is not None:
            current_duration = time.time() - event_start_time
            alert_text = f"+ {current_duration:.1f}s"
            color_alert = (0, 0, 255) if current_duration > 2.0 else (0, 165, 255)
            cv2.putText(frame, alert_text, (140, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_alert, 1)



        # ==========================================
        # [LOGIC 1] ĐẾM THỜI GIAN VÀ BẮT ĐẦU SỰ KIỆN KHẨN CẤP
        # ==========================================
        if state != "Normal" and state != "Talking":
            if current_event_type != state: # Trạng thái mới bắt đầu
                current_event_type = state
                event_start_time = time.time()
                alert_sent = False
                
                # In log khi bắt đầu trạng thái nguy hiểm
                if state == "Drowsy": print("\n[NGUY HIỂM] Có dấu hiệu nhắm mắt (Drowsy)!")
                elif state == "Distracted": print("\n[CẢNH BÁO] Mất tập trung (Distracted)!")
                elif state == "Yawning": print("\n[NHẮC NHỞ] Đang ngáp (Yawning)!")

            if event_start_time is not None:
                # Tính số giây đã trôi qua kể từ lúc bắt đầu nhắm mắt/ngáp/mất tập trung
                duration_seconds = time.time() - event_start_time
                
                # Cảnh báo KHẨN CẤP nếu ngủ gục > 2s hoặc ngáp > 3s (ví dụ)
                if current_event_type == "Drowsy" and duration_seconds > 2.0 and not alert_sent:
                    severity = "emergency" if duration_seconds > 4.0 else "danger"
                    
                    # Tạo tên file kèm mốc thời gian (thử mục temp_alert)
                    filename = f"temp_alert/images/alert_drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    vid_filename = f"temp_alert/videos/alert_drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    cv2.imwrite(filename, frame)
                    print(f">>> [LƯU ẢNH CAMERA] Lưu file {filename} thành công. Đã ngủ: {duration_seconds:.1f}s")
                    
                    # Nhánh Thread xử lý song song Upload và Xuất Video
                    frames_to_save = list(frame_buffer)
                    threading.Thread(target=process_and_upload_alert, args=(
                        trip_id, user_id, "eyes_closed", severity, duration_seconds, filename, frames_to_save, vid_filename
                    )).start()
                    alert_sent = True # Chốt, không gửi lặp lại gây spam
                    
                # Gửi cảnh báo mất tập trung nếu trên 4 giây
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
            # TÀI XẾ ĐÃ TRỞ LẠI BÌNH THƯỜNG
            if current_event_type != "Normal":
                duration = time.time() - event_start_time if event_start_time else 0
                print(f"\n[THÔNG TIN] Tài xế đã tỉnh / Tập trung lại. Tổng t.gian sự kiện vừa rồi: {duration:.1f}s")
                
                # Cộng dồn thời gian và số lần vào biến Analytics Trip
                if current_event_type == "Drowsy":
                    total_eye_closed_time += duration
                elif current_event_type == "Yawning":
                    total_yawn_count += 1
                elif current_event_type == "Distracted":
                    total_head_tilt_count += 1
                    
                # Reset lại bộ đếm sự kiện
                current_event_type = "Normal"
                event_start_time = None
                alert_sent = False
                
        # ==========================================
        # [LOGIC 2] GỬI BẢN BÁO CÁO ĐỊNH KỲ (ANALYTICS)
        # ==========================================
        # Giả định: Cứ sau 1 phút sẽ in tổng kết (nếu real thì khoảng 5-10 phút đẩy DB 1 lần tùy cấu hình băng thông)
        if (time.time() - last_sync_time) > 60: 
            print(f"\n=== [SUPABASE-ANALYTICS] GỌI API ĐỒNG BỘ BẢNG TRIPS ({trip_id}) ===")
            threading.Thread(target=update_trip_analytics, args=(trip_id, total_yawn_count, total_head_tilt_count, total_eye_closed_time)).start()
            last_sync_time = time.time()
            
        previous_state = state

    # Đưa frame đã vẽ GIỮA CHỪNG TRƯỚC ĐÓ vào bộ nhớ đệm deque
    frame_buffer.append(frame.copy())

    try:
        cv2.imshow("Smart AI DMS Architecture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except Exception as e:
        print(f"[UI-ERROR] Loi hiển thị: {e}")

cv2.destroyAllWindows()
vs.stop()