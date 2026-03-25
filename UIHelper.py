import cv2
from datetime import datetime

class UIHelper:
    """
    Lớp hỗ trợ hiển thị thông tin trực quan lên khung hình (Frame Overlay).
    """
    def draw_status(self, frame, state, color):
        """
        Hiển thị trạng thái AI hiện tại (Normal, Drowsy, Distracted, Yawning).
        """
        cv2.putText(frame, f"STATE: {state}", (140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_analytics(self, frame, yawns, head_tilts, eye_time, pitch):
        """
        Hiển thị các thông số tích lũy của hành trình trên góc phải màn hình.
        - yawns: Số lần ngáp
        - head_tilts: Số lần nghiêng đầu/mất tập trung
        - eye_time: Tổng thời gian nhắm mắt
        - pitch: Góc cúi đầu hiện tại
        """
        cv2.putText(frame, f"Y: {yawns} ", (430, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"D: {head_tilts}", (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"E: {eye_time:.1f}s", (430, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"P: {pitch:.1f}deg", (430, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    def draw_clock(self, frame):
        """
        Hiển thị đồng hồ thời gian thực ở góc dưới bên phải.
        """
        current_dt = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        cv2.putText(frame, current_dt, (330, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_calibration_progress(self, frame, progress):
        """
        Hiển thị thanh tiến trình khi hệ thống đang học khuôn mặt driver (Calibration).
        """
        cv2.putText(frame, f"LEARNING: {progress*100:.1f}%", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    def draw_warning_text(self, frame, text, color=(0,0,255)):
        """
        Hiển thị các văn bản cảnh báo ở giữa màn hình.
        """
        cv2.putText(frame, text, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
