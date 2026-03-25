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
        # Tải các biến môi trường từ file .env
        load_dotenv()
        self.user_id = os.getenv("USER_ID", "00000000-0000-0000-0000-000000000000")
        self.trip_id = os.getenv("TRIP_ID", "11111111-1111-1111-1111-111111111111")
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        
        try:
            # Khởi tạo Supabase Client
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("[INFO] Kết nối Supabase thành công.")
        except Exception:
            self.client = None
            print("[WARNING] Supabase chưa được cấu hình. Chạy ở chế độ Offline.")

    def update_trip_analytics(self, yawns, head_tilts, eye_time):
        """
        Gửi dữ liệu thống kê tổng hợp của chuyến đi về hệ thống.
        - yawns: Số lần ngáp.
        - head_tilts: Số lần mất tập trung/nghiêng đầu.
        - eye_time: Tổng thời gian nhắm mắt (giây).
        """
        if not self.client: return
        try:
            # Đánh giá mức độ mệt mỏi dựa trên các thông số
            level = "Safe"
            if eye_time > 15.0 or yawns > 10: level = "Khẩn cấp"
            elif eye_time > 5.0 or yawns > 3 or head_tilts > 5: level = "Nguy hiểm"
            
            payload = {
                "total_yawn_count": yawns,
                "total_eye_closed_time": float(eye_time),
                "total_head_tilt": head_tilts,
                "fatigue_level": level
            }
            # Cập nhật bảng trips trong Supabase
            self.client.table("trips").update(payload).eq("id", self.trip_id).execute()
            print(f"-> [SUPABASE] Đồng bộ trips thành công.")
        except Exception as e:
            print(f"[ERROR] Sync Trip: {e}")

    def upload_alert(self, event_type, severity, duration, image_path, video_path):
        """
        Tải các phương tiện cảnh báo (ảnh, video) lên Storage và lưu thông tin vào bảng alerts.
        """
        if not self.client: return
        try:
            # 1. Tải tệp tin lên Storage bucket 'alerts_media'
            self.client.storage.from_("alerts_media").upload(image_path, image_path)
            self.client.storage.from_("alerts_media").upload(video_path, video_path)
            
            # 2. Lấy URL công khai của các tệp tin đã tải lên
            img_url = self.client.storage.from_("alerts_media").get_public_url(image_path)
            vid_url = self.client.storage.from_("alerts_media").get_public_url(video_path)

            # 3. Chèn bản ghi dữ liệu vào bảng SQL 'alerts'
            row = {
                "trip_id": self.trip_id,
                "user_id": self.user_id,
                "alert_type": event_type,
                "severity": severity,
                "duration_seconds": float(duration),
                "image_url": img_url,
                "video_url": vid_url,
                "created_at": datetime.now().isoformat()
            }
            self.client.table("alerts").insert(row).execute()
            print(f"[SUPABASE] Gửi cảnh báo {event_type} thành công.")
        except Exception as e:
            print(f"[ERROR] Upload Alert API: {e}")
