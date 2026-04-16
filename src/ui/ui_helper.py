import cv2
import time
from datetime import datetime

class UIHelper:
    """
    Lớp hỗ trợ hiển thị thông tin trực quan lên khung hình (Frame Overlay).
    """
    def __init__(self):
        # === CACHE CHO ĐỒNG HỒ ===
        # Lưu chuỗi thời gian đã format để tránh gọi datetime.now() + strftime() mỗi frame
        self._cached_clock_text = ""
        # Thời điểm cập nhật lần cuối (epoch seconds)
        self._last_clock_update = 0

    def draw_status(self, frame, state, color):
        """
        Hiển thị trạng thái AI hiện tại (Normal, Drowsy, Distracted, Yawning).
        """
        cv2.putText(frame, f"STATE: {state}", (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_analytics(self, frame, yawns, head_tilts, eye_time, pitch):
        """
        Hiển thị các thông số tích lũy của hành trình trên góc phải màn hình.
        - yawns: Số lần ngáp
        - head_tilts: Số lần nghiêng đầu/mất tập trung
        - eye_time: Tổng thời gian nhắm mắt
        - pitch: Góc cúi đầu hiện tại
        """
        cv2.putText(frame, f"Y: {yawns} ",       (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"D: {head_tilts}",   (5, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"E: {eye_time:.1f}s",(5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"P: {pitch:.1f}deg", (5, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    def draw_clock(self, frame):
        """
        Hiển thị đồng hồ thời gian thực ở góc dưới bên phải.
        Sử dụng cache để tránh gọi strftime() mỗi frame (tiết kiệm CPU).
        """
        now = time.time()
        # Chỉ cập nhật chuỗi thời gian khi đã qua ít nhất 1 giây
        if now - self._last_clock_update >= 1.0:
            self._cached_clock_text = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            self._last_clock_update = now
        cv2.putText(frame, self._cached_clock_text, (5, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_calibration_progress(self, frame, progress):
        """
        Hiển thị thanh tiến trình khi hệ thống đang học khuôn mặt driver (Calibration).
        - progress: giá trị từ 0.0 đến 1.0
        """
        cv2.putText(frame, f"LEARNING: {progress*100:.1f}%", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    def draw_warning_text(self, frame, text, color=(0,0,255)):
        """
        Hiển thị các văn bản cảnh báo ở giữa màn hình.
        """
        cv2.putText(frame, text, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
