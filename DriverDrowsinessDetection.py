#!/usr/bin/env python
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import threading
from collections import deque

# --- IMPORT CÁC MODULE CUSTOM CŨ ---
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Calibration import Calibrator
from PredictMaker import DecisionMaker

# --- IMPORT CÁC MODULE MỚI SAU KHI REFACTOR ---
from Backend import BackendManager
from AlertHandler import AlertHandler
from UIHelper import UIHelper

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
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./dlib_shape_predictor/dlib_face_recognition_resnet_model_v1.dat')

# Cấu hình logic AI DMS
calibrator = Calibrator(required_frames=100) 
decision_maker = DecisionMaker(window_size=30, model_path="Models/dms_model_int8.tflite")

# Tham số luồng video
frame_width, frame_height = 480, 480
image_points = np.zeros((6, 2), dtype="double")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mStart, mEnd = 49, 68

# Các biến quản lý thời gian
frame_count = 0
last_sync_time = time.time()
last_calibration_time = time.time()
calibration_interval = 60 # Tự động hiệu chuẩn lại sau mỗi 60 giây

# Buffer lưu trữ frame để xuất video cảnh báo (khoảng 2-4 giây)
frame_buffer = deque(maxlen=60)

# Khởi động Camera
print("[INFO] Khởi động hệ thống Driver Monitoring System (DMS)...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

try:
    while True:
        # --- BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ FRAME ---
        frame = vs.read()
        if frame is None: break
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        # Hiển thị đồng hồ thời gian thực
        ui.draw_clock(frame)
        
        # --- BƯỚC 2: QUẢN LÝ ĐỊNH DANH DRIVER (IDENTIFICATION) ---
        driver_rect = None
        if not calibrator.is_calibrated:
            if len(rects) == 1: driver_rect = rects[0]
            else: ui.draw_warning_text(frame, "WAITING FOR DRIVER...")
        else:
            if len(rects) == 1: driver_rect = rects[0]
            elif len(rects) > 1:
                # Chỉ kiểm tra Face ID mỗi 30 frame để giảm tải CPU
                if frame_count % 30 == 0:
                    for r in rects:
                        shape_tmp = predictor(gray, r)
                        enc = np.array(face_encoder.compute_face_descriptor(frame, shape_tmp))
                        if calibrator.is_driver(enc): driver_rect = r; break
                else: driver_rect = rects[0]

        # --- BƯỚC 3: XỬ LÝ KHÍ KHÔNG THẤY KHUÔN MẶT (DISTRACTED) ---
        if driver_rect is None:
            if calibrator.is_calibrated:
                # Ghi nhận trạng thái mất tập trung (Distracted) khi không thấy mặt
                alert_handler.process_state("Distracted", frame, frame_buffer)
                if alert_handler.current_event == "Distracted":
                    dur = time.time() - alert_handler.start_time
                    ui.draw_warning_text(frame, f"NO FACE: {dur:.1f}s")
            
            # Cập nhật buffer và tiếp tục vòng lặp
            frame_buffer.append(frame.copy())
            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            continue
        
        # --- BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT ---
        shape_obj = predictor(gray, driver_rect)
        shape = face_utils.shape_to_np(shape_obj)
        
        # Tính toán EAR (Mắt) và MAR (Miệng)
        ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0
        mar = mouth_aspect_ratio(shape[mStart:mEnd])
        
        # Trích xuất góc cúi đầu (Head Pose) từ các điểm mốc
        for idx, lm in enumerate([33, 8, 36, 45, 48, 54]):
            image_points[idx] = shape[lm]
        (h_deg, y_deg, p_raw, start_p, end_p, end_p2) = getHeadTiltAndCoords(gray.shape, image_points, frame_height)
        pitch = h_deg[0] if h_deg else 0.0

        # --- BƯỚC 5: PHÂN LOẠI TRẠNG THÁI AI (CALIBRATION HOẶC PREDICTION) ---
        display_state, display_color = "Normal", (0,255,0)
        
        if not calibrator.is_calibrated:
            # Giai đoạn HIỆU CHUẨN: Học các chỉ số mắt/miệng/đầu bình thường của Driver
            calibrator.update(ear, mar, p_raw)
            calibrator.update_face(np.array(face_encoder.compute_face_descriptor(frame, shape_obj)))
            ui.draw_calibration_progress(frame, calibrator.get_progress())
        else:
            # Giai đoạn NHẬN DIỆN: Đưa thông số vào mô hình AI để dự đoán trạng thái
            decision_maker.update_buffer(ear, mar, pitch, y_deg, p_raw, 
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
        if (time.time() - last_sync_time) > 60:
            # Chạy thread ngầm để không làm trễ Camera
            threading.Thread(target=backend.update_trip_analytics, 
                             args=(alert_handler.total_yawn_count, 
                                   alert_handler.total_head_tilt_count, 
                                   alert_handler.total_eye_closed_time)).start()
            last_sync_time = time.time()
            
            # Tự động hiệu chuẩn lại sau một khoảng thời gian
            if calibrator.is_calibrated and (time.time() - last_calibration_time > calibration_interval):
                calibrator.reset()
                last_calibration_time = time.time()
                print("[INFO] Đang thực hiện Re-Calibration...")

        # Lưu frame vào buffer và hiển thị lên cửa sổ chính
        frame_buffer.append(frame.copy())
        cv2.imshow("DMS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    # --- GIẢI PHÓNG TÀI NGUYÊN ---
    print("[INFO] Đang dừng hệ thống...")
    cv2.destroyAllWindows()
    vs.stop()