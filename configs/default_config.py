"""
Cấu hình mặc định của hệ thống Driver Monitoring System (DMS).
Tập trung toàn bộ các ngưỡng và đường dẫn vào một nơi duy nhất
để dễ dàng tinh chỉnh mà không cần mở nhiều file.
"""

# ============================================================
# ĐƯỜNG DẪN MÔ HÌNH (Model Paths)
# ============================================================
MODEL_TFLITE_PATH = "models/tflite/dms_model_int8.tflite"
MODEL_DLIB_LANDMARK = "models/dlib/shape_predictor_68_face_landmarks.dat"
MODEL_DLIB_RECOGNITION = "models/dlib/dlib_face_recognition_resnet_model_v1.dat"

# ============================================================
# ĐƯỜNG DẪN ÂM THANH (Audio Paths)
# ============================================================
AUDIO_DROWSY = "assets/audio/drowsy.mp3"
AUDIO_YAWNING = "assets/audio/yawning.mp3"
AUDIO_DISTRACTED = "assets/audio/distracted.mp3"
AUDIO_CALIBRATION = "assets/audio/calibration.mp3"

# ============================================================
# CẤU HÌNH CAMERA
# ============================================================
FRAME_WIDTH = 320          # Chiều rộng frame (pixels) - tối ưu cho Jetson Nano
FRAME_HEIGHT = 320         # Chiều cao frame (pixels)
CAMERA_SOURCE = 0          # Chỉ số camera (0 = mặc định, 1 = USB ngoài)

# ============================================================
# CẤU HÌNH CALIBRATION
# ============================================================
CALIBRATION_REQUIRED_FRAMES = 100  # Số frame cần thu thập để hoàn thành calibration

# ============================================================
# CẤU HÌNH AI INFERENCE
# ============================================================
DECISION_WINDOW_SIZE = 15   # Kích thước cửa sổ trượt (frames) cho PredictMaker
SYNC_INTERVAL = 30          # Khoảng thời gian đồng bộ backend (giây)
FRAME_BUFFER_SIZE = 60      # Số frame tối đa lưu trong buffer video (~2-4 giây)

# ============================================================
# NGƯỠNG CẢNH BÁO (Alert Thresholds)
# Thời gian tối thiểu (giây) để xác nhận và gửi cảnh báo
# ============================================================
ALERT_THRESHOLDS = {
    "Yawning":    1.0,   # Ngáp liên tục > 1.0s
    "Drowsy":     1.5,   # Mắt nhắm > 1.5s
    "Distracted": 2.0,   # Mất tập trung > 2.0s
}

# ============================================================
# CẤU HÌNH FRAME SKIPPING
# ============================================================
FRAME_SKIP_INTERVAL = 3  # Cứ 1 frame phát hiện khuôn mặt, (FRAME_SKIP_INTERVAL-1) frame dùng cache
