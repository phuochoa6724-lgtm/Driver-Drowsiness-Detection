#!/usr/bin/env python
"""
Driver Monitoring System (DMS) - Điểm khởi động chính của ứng dụng.
Cấu trúc dự án đã được tổ chức lại theo chuẩn Python package:
  - src/core/     : Backend, Calibration, AlertHandler
  - src/detection/: EAR, MAR, HeadPose
  - src/inference/: PredictMaker (TFLite AI)
  - src/ui/       : UIHelper
"""
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import threading
from collections import deque

# --- IMPORT CÁC MODULE TÙY CHỈNH (cấu trúc mới) ---
from src.detection.ear import eye_aspect_ratio
from src.detection.mar import mouth_aspect_ratio
from src.detection.head_pose import getHeadTiltAndCoords
from src.core.calibration import Calibrator
from src.inference.predict_maker import DecisionMaker
from src.core.backend import BackendManager
from src.core.alert_handler import AlertHandler
from src.ui.ui_helper import UIHelper

# --- KHỞI TẠO CÁC THÀNH PHẦN HỆ THỐNG ---
# BackendManager: Quản lý kết nối Supabase và lưu trữ dữ liệu
backend = BackendManager()
# AlertHandler: Quản lý logic các sự kiện cảnh báo (Drowsy, Distracted...) và video chứng cứ
alert_handler = AlertHandler(backend)
# UIHelper: Hỗ trợ vẽ các thông tin trạng thái và số liệu lên màn hình Camera
ui = UIHelper()

# --- KHỞI TẠO MÔ HÌNH AI & PHẦN CỨNG ---
print("[INFO] Đang nạp các mô hình AI và nhận diện khuôn mặt...")
detector = dlib.get_frontal_face_detector()
# Đường dẫn model dlib đã cập nhật sang models/dlib/
predictor = dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('models/dlib/dlib_face_recognition_resnet_model_v1.dat')

# Cấu hình logic AI DMS
calibrator = Calibrator(required_frames=100) 
# Cài đặt window_size là 15 (tương đương ~2-3 giây trễ trên Jetson Nano)
# giúp phản hồi cập nhật các trạng thái (EAR,...) nhanh nhạy hơn.
decision_maker = DecisionMaker(window_size=15, model_path="models/tflite/dms_model_int8.tflite")

# Tham số kích thước luồng video (sử dụng 320x320 để tối ưu tốc độ xử lý trên thiết bị nhúng)
frame_width, frame_height = 320, 320
image_points = np.zeros((6, 2), dtype="double")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mStart, mEnd = 49, 68

# Các biến quản lý thời gian
frame_count = 0
last_sync_time = time.time()
calibration_interval = 60 # Khoảng thời gian cho hiệu chuẩn lại (Hiện tại vô hiệu hóa theo yêu cầu dự án)
calibration_voice_played = False # Cờ kiểm soát việc phát âm thanh nhắc nhở

# Buffer lưu trữ frame để xuất video cảnh báo (khoảng 2-4 giây)
frame_buffer = deque(maxlen=60)

# Khởi động Camera
print("[INFO] Khởi động hệ thống Driver Monitoring System (DMS)...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Khởi tạo biến theo dõi khuôn mặt tài xế
last_driver_rect = None
cached_rects = [] # Bộ nhớ tạm lưu khung mặt cho Frame Skipping

try:
    while True:
        # --- BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ FRAME ---
        frame = vs.read()
        if frame is None: break
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # CƠ CHẾ FRAME SKIPPING - Giảm tải CPU
        # Cứ 1 frame tiến hành phát hiện khuôn mặt, 2 frame tiếp theo sẽ dùng lại kết quả cũ
        if frame_count % 3 == 1 or len(cached_rects) == 0:
            rects = detector(gray, 0)
            cached_rects = rects
        else:
            rects = cached_rects
        
        # Hiển thị đồng hồ thời gian thực
        ui.draw_clock(frame)
        
        # --- BƯỚC 2: QUẢN LÝ ĐỊNH DANH DRIVER (IDENTIFICATION) ---
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
                # Tránh nhận nhầm người ngồi cạnh bằng cách chọn khuôn mặt gần vị trí tài xế cũ nhất
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

        # --- BƯỚC 3: XỬ LÝ KHI KHÔNG THẤY KHUÔN MẶT (DISTRACTED) ---
        if driver_rect is None:
            if calibrator.is_calibrated:
                # Ghi nhận trạng thái mất tập trung (Distracted) khi không thấy mặt
                alert_handler.process_state("Distracted", frame, frame_buffer)
                if alert_handler.current_event == "Distracted":
                    ui.draw_status(frame, "Distracted", (0, 0, 255))
            
            # Cập nhật buffer và tiếp tục vòng lặp
            frame_buffer.append(frame.copy())
            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            continue
        
        # --- BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT ---
        shape_obj = predictor(gray, driver_rect)
        shape = face_utils.shape_to_np(shape_obj)
        
        # Tính toán EAR (Mắt) và MAR thô (Miệng)
        ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0
        mar = mouth_aspect_ratio(shape[mStart:mEnd])
        
        # Lấy tọa độ 6 điểm mốc tĩnh (nửa trên): đầu mũi, gốc mũi, 4 khóe mắt
        # Không dùng cằm và mép miệng để tránh sai lệch khi tài xế há miệng/ngáp
        for idx, lm in enumerate([30, 27, 36, 45, 39, 42]):
            image_points[idx] = shape[lm]
        (h_deg, y_deg, p_raw, start_p, end_p, end_p2) = getHeadTiltAndCoords(gray.shape, image_points, frame_height)
        pitch = h_deg[0] if len(h_deg) > 0 else 0.0

        # BÙ MAR THEO GÓC YAW (Yaw Compensation):
        # Khi đầu quay ngang góc θ, chiều ngang miệng bị co lại theo cos(θ),
        # làm MAR tăng ảo (miệng trông như đang há dù thực tế không há).
        # Nhân mar × cos(|yaw|) để triệt tiêu hiệu ứng phối cảnh (perspective foreshortening).
        yaw_rad = abs(y_deg) * np.pi / 180.0
        mar_corrected = mar * float(np.cos(yaw_rad))

        # --- BƯỚC 5: PHÂN LOẠI TRẠNG THÁI AI (CALIBRATION HOẶC PREDICTION) ---
        display_state, display_color = "Normal", (0,255,0)
        
        if not calibrator.is_calibrated:
            # Phát âm thanh nhắc nhở lần đầu tiên khi bắt đầu calibration
            if not calibration_voice_played:
                alert_handler.play_calibration_reminder()
                calibration_voice_played = True
                
            # Giai đoạn HIỆU CHUẨN: Học các chỉ số mắt/miệng/đầu bình thường của Driver
            # Sử dụng mar_corrected (sau khi bù yaw) để baseline cũng được chuẩn hoá theo góc nhìn thẳng
            calibrator.update(ear, mar_corrected, p_raw)
            calibrator.update_face(np.array(face_encoder.compute_face_descriptor(frame, shape_obj)))
            ui.draw_calibration_progress(frame, calibrator.get_progress())
        else:
            # Giai đoạn NHẬN DIỆN: Đưa thông số vào mô hình AI để dự đoán trạng thái
            # Sử dụng mar_corrected thay cho mar thô để tránh nhẬn nhầm Yawning khi quay đầu
            decision_maker.update_buffer(ear, mar_corrected, pitch, y_deg, p_raw, 
                                        calibrator.ear_baseline, 
                                        calibrator.mar_baseline, 
                                        calibrator.pitch_raw_baseline)
            display_state = decision_maker.predict_state()
            
            # Cập nhật màu sắc UI theo mức độ nghiêm trọng
            if display_state in ["Drowsy", "Distracted"]: display_color = (0,0,255)
            elif display_state == "Yawning": display_color = (0,165,255)

        # --- BƯỚC 6: QUẢN LÝ CẢNH BÁO & GHI HÌNH ---
        alert_handler.process_state(display_state, frame, frame_buffer)

        # --- BƯỚC 7: HIỂN THỊ THÔNG TIN UI OVERLAY ---
        ui.draw_status(frame, display_state, display_color)
        ui.draw_analytics(frame, 
                         alert_handler.total_yawn_count, 
                         alert_handler.total_head_tilt_count, 
                         alert_handler.total_eye_closed_time, 
                         pitch)
        
        # --- BƯỚC 8: ĐỒNG BỘ DỮ LIỆU VỚI BACKEND (ĐỊNH KỲ) ---
        if (time.time() - last_sync_time) > 30:
            # Chạy thread ngầm để không làm trễ Camera
            threading.Thread(target=backend.update_trip_analytics, 
                             args=(alert_handler.total_yawn_count, 
                                   alert_handler.total_head_tilt_count, 
                                   alert_handler.total_eye_closed_time,
                                   alert_handler.total_drowsy_count,
                                   alert_handler.total_distracted_count)).start()
            last_sync_time = time.time()

        # Lưu frame vào buffer và hiển thị lên cửa sổ chính
        frame_buffer.append(frame.copy())
        cv2.imshow("DMS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    # --- GIẢI PHÓNG TÀI NGUYÊN ---
    print("[INFO] Đang dừng hệ thống và đóng hành trình...")
    # Gửi dữ liệu cuối cùng chuẩn bị kết thúc
    backend.close_session(alert_handler.total_yawn_count, 
                          alert_handler.total_head_tilt_count, 
                          alert_handler.total_eye_closed_time,
                          alert_handler.total_drowsy_count,
                          alert_handler.total_distracted_count)
                          
    cv2.destroyAllWindows()
    vs.stop()