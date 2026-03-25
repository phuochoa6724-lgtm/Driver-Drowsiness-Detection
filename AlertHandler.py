import cv2
import os
import time
import threading
from datetime import datetime

class AlertHandler:
    """
    Lớp quản lý các sự kiện cảnh báo và ghi lại video/ảnh bằng chứng.
    Hỗ trợ kiểm tra ngưỡng thời gian và lưu video ra file cục bộ.
    """
    def __init__(self, backend):
        self.backend = backend
        # Tạo thư mục lưu tạm cho ảnh và video cảnh báo
        os.makedirs("temp_alert/images", exist_ok=True)
        os.makedirs("temp_alert/videos", exist_ok=True)
        
        # Trạng thái cảnh báo hiện tại (Mặc định là Normal)
        self.current_event = "Normal"
        self.start_time = None
        self.alert_sent = False
        
        # Thống kê tích lũy của toàn bộ hành trình
        self.total_yawn_count = 0 
        self.total_head_tilt_count = 0
        self.total_eye_closed_time = 0.0

    def process_state(self, state, frame, frame_buffer):
        """
        Xử lý trạng thái từ AI trả về để quản lý vòng đời sự kiện (Bắt đầu -> Gửi cảnh báo -> Kết thúc).
        """
        if state not in ["Normal", "Talking"]:
            # Nếu bắt đầu một trạng thái cảnh báo mới (Drowsy, Distracted, Yawning)
            if self.current_event != state:
                self.current_event = state
                self.start_time = time.time()
                self.alert_sent = False
                print(f"\n[!] BẮT ĐẦU SỰ KIỆN: {state}")
            
            # Tính toán thời gian đã diễn ra của sự kiện hiện tại
            duration = time.time() - self.start_time
            
            # Hiển thị thời gian cảnh báo đang chạy lên Frame
            cv2.putText(frame, f"+{duration:.1f}s", (140, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)

            # Kiểm tra ngưỡng để gửi cảnh báo (Drowsy > 1.5s, các loại khác > 3.0s)
            limit = 1.5 if state == "Drowsy" else 3.0
            if duration > limit and not self.alert_sent:
                # Kích hoạt tiến trình lưu và gửi cảnh báo
                self._trigger_alert(state, duration, frame, list(frame_buffer))
                self.alert_sent = True
        else:
            # Khi trạng thái trở về bình thường, kết thúc sự kiện hiện tại
            if self.current_event != "Normal":
                duration = time.time() - self.start_time
                print(f"[#] KẾT THÚC SỰ KIỆN: {self.current_event} ({duration:.1f}s)")
                
                # Cập nhật số liệu tích lũy tùy theo loại sự kiện
                if self.current_event == "Drowsy": self.total_eye_closed_time += duration
                elif self.current_event == "Yawning": self.total_yawn_count += 1
                elif self.current_event == "Distracted": self.total_head_tilt_count += 1
                
                # Reset các biến trạng thái
                self.current_event = "Normal"
                self.start_time = None
                self.alert_sent = False

    def _trigger_alert(self, event_type, duration, current_frame, frames):
        """Kích hoạt việc lưu file và gửi lên backend (chạy ngầm)."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_p = f"temp_alert/images/A_{ts}.jpg"
        vid_p = f"temp_alert/videos/V_{ts}.mp4"
        
        # Lưu ảnh bằng chứng (Frame hiện tại)
        cv2.imwrite(img_p, current_frame)
        
        # Chạy luồng xử lý video (MP4) và upload ngầm để không gây giật lag Camera
        threading.Thread(target=self._save_and_upload, args=(event_type, duration, frames, img_p, vid_p)).start()

    def _save_and_upload(self, event_type, duration, frames, img_p, vid_p):
        """Lưu video MP4 từ danh sách frame và yêu cầu backend tải lên Supabase."""
        if frames:
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_p, fourcc, 15.0, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            
        # Gọi backend để thực hiện upload ảnh, video và lưu database
        self.backend.upload_alert(event_type.lower(), "danger", duration, img_p, vid_p)
