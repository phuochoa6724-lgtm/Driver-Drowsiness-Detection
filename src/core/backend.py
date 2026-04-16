import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

class BackendManager:
    """
    Lớp quản lý kết nối và tương tác với Supabase Backend.
    Hỗ trợ đồng bộ dữ liệu hành trình và tải lên các cảnh báo (ảnh/video).
    """
    def __init__(self):
        # Tải các biến môi trường từ file .env (tìm kiếm từ thư mục gốc dự án)
        load_dotenv()
        self.user_id = os.getenv("USER_ID", " ")
        self.trip_id = os.getenv("TRIP_ID", " ")
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        
        try:
            # Khởi tạo Supabase Client
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("[INFO] Kết nối Supabase thành công.")
            
            # Khởi tạo Trip ID mới mỗi lần chạy
            self._initialize_session()
        except Exception as e:
            self.client = None
            print(f"[WARNING] Supabase chưa được cấu hình hoặc lỗi: {e}. Chạy ở chế độ Offline.")

    def _initialize_session(self):
        """Khởi tạo User Profile và Trip mới để đảm bảo tính toàn vẹn dữ liệu."""
        if not self.client: return
        try:
            # Tạo bản ghi Trip mới với thời gian bắt đầu hiện tại
            self.session_start = datetime.now()
            new_trip_payload = {
                "user_id": self.user_id,
                "start_time": self.session_start.isoformat(),
                "status": "Đang chạy"
            }
            res = self.client.table("trips").insert(new_trip_payload).execute()
            
            # Ghi nhận trip_id mới được tạo tự động bởi Supabase
            if res.data:
                self.trip_id = res.data[0]['id']
                print(f"[SUCCESS] Đã khởi tạo phiên làm việc mới. Trip ID: {self.trip_id}")
        except Exception as e:
            print(f"[WARNING] Không thể tạo Trip mới tự động: {e}. Sử dụng ID mặc định.")

    def _get_metrics_payload(self, yawns, head_tilts, eye_time, drowsy_count, distracted_count=0):
        """Hàm trợ giúp để tính toán mức độ mệt mỏi và đóng gói dữ liệu."""
        # Xác định cấp độ nguy hiểm dựa trên ngưỡng cảnh báo
        level = "An toàn"
        if eye_time > 15.0 or yawns > 10: level = "Khẩn cấp"
        elif eye_time > 3.0 or yawns > 2 or head_tilts > 3 or distracted_count > 3: level = "Cảnh báo"
        
        # Đóng gói dữ liệu số liệu hành trình (payload) để chuẩn bị đồng bộ
        return {
            "total_yawn_count": int(yawns),
            "total_head_tilt": int(head_tilts),
            "total_eye_closed_time": float(eye_time),
            "total_alerts": int(yawns + head_tilts + drowsy_count + distracted_count),
            "fatigue_level": level
        }

    def update_trip_analytics(self, yawns, head_tilts, eye_time, drowsy_count, distracted_count=0):
        """Đồng bộ dữ liệu thống kê định kỳ lên Supabase."""
        if not self.client: return
        try:
            payload = self._get_metrics_payload(yawns, head_tilts, eye_time, drowsy_count, distracted_count)
            self.client.table("trips").update(payload).eq("id", self.trip_id).execute()
            print(f"[*] [SUPABASE] Đồng bộ metrics thành công.")
        except Exception as e:
            print(f"[ERROR] Sync Trip: {e}")

    def close_session(self, yawns, head_tilts, eye_time, drowsy_count, distracted_count=0):
        """Kết thúc hành trình: Cập nhật end_time và trạng thái cuối cùng."""
        if not self.client or not hasattr(self, 'session_start'): return
        try:
            end_time = datetime.now()
            duration_minutes = int((end_time - self.session_start).total_seconds() / 60)
            
            payload = self._get_metrics_payload(yawns, head_tilts, eye_time, drowsy_count, distracted_count)
            payload.update({
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
                "status": "Đã kết thúc"
            })
            
            self.client.table("trips").update(payload).eq("id", self.trip_id).execute()
            print(f"[*] [SUPABASE] Đã đóng hành trình ({duration_minutes} phút).")
        except Exception as e:
            print(f"[ERROR] Close Session: {e}")

    def upload_alert(self, event_type, severity, duration, video_path):
        """
        Tải video cảnh báo lên kho lưu trữ Storage và lưu thông tin vào bảng alerts.
        """
        if not self.client: return
        try:
            # 1. Tải video lên Storage bucket 'alerts_media'
            self.client.storage.from_("alerts_media").upload(video_path, video_path)
            
            # 2. Lấy URL công khai của video đã tải lên
            vid_url = self.client.storage.from_("alerts_media").get_public_url(video_path)

            # 3. Chèn bản ghi dữ liệu vào bảng SQL 'alerts'
            row = {
                "trip_id": self.trip_id,
                "user_id": self.user_id,
                "alert_type": event_type,
                "severity": severity,
                "duration_seconds": float(duration),
                "video_url": vid_url,
                "created_at": datetime.now().isoformat()
            }
            self.client.table("alerts").insert(row).execute()
            print(f"[SUPABASE] Gửi cảnh báo {event_type} thành công.")
        except Exception as e:
            print(f"[ERROR] Upload Alert API: {e}")
