import os
import csv
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

class BackendManager:
    """
    Lớp quản lý kết nối và tương tác với Supabase Backend.
    Hỗ trợ đồng bộ dữ liệu hành trình và tải lên các cảnh báo (ảnh/video).
    Hỗ trợ lưu trữ offline tự động vào file CSV trong thư mục logs.
    """
    def __init__(self):
        # Đảm bảo thư mục logs tồn tại
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.offline_trips_file = os.path.join(self.log_dir, "offline_trips.csv")
        self.offline_alerts_file = os.path.join(self.log_dir, "offline_alerts.csv")
        
        # Tạo file CSV với tiêu đề nếu chưa tồn tại
        self._init_offline_logs()

        # Tải các biến môi trường từ file .env (tìm kiếm từ thư mục gốc dự án)
        load_dotenv()
        self.user_id = os.getenv("USER_ID", "local_user")
        
        # Lấy datetime hiện tại làm suffix cho offline trip_id để tránh trùng lặp
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trip_id = os.getenv("TRIP_ID", f"offline_trip_{current_time_str}")
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        
        try:
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Thiếu URL hoặc KEY của Supabase")
            # Khởi tạo Supabase Client
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("[INFO] Kết nối Supabase thành công.")
        except Exception as e:
            self.client = None
            print(f"[WARNING] Supabase chưa được cấu hình hoặc lỗi: {e}. Chạy hoàn toàn ở chế độ Offline.")
        
        # Khởi tạo Trip ID mới mỗi lần chạy (Cả online và offline)
        self._initialize_session()

    def _init_offline_logs(self):
        """Khởi tạo cấu trúc file log CSV nếu chưa có"""
        if not os.path.exists(self.offline_trips_file):
            with open(self.offline_trips_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["time", "action", "trip_id", "user_id", "status", "yawn_count", "head_tilt", "eye_closed_time", "alerts", "fatigue_level", "duration_minutes"])
        
        if not os.path.exists(self.offline_alerts_file):
            with open(self.offline_alerts_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["time", "trip_id", "user_id", "alert_type", "severity", "duration_seconds", "video_path"])

    def _log_trip_offline(self, action, status, payload, duration_minutes=""):
        """Ghi nhận dữ liệu trip vào file CSV (Bằng chứng offline)"""
        try:
            with open(self.offline_trips_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    action,
                    self.trip_id,
                    self.user_id,
                    status,
                    payload.get("total_yawn_count", 0),
                    payload.get("total_head_tilt", 0),
                    payload.get("total_eye_closed_time", 0.0),
                    payload.get("total_alerts", 0),
                    payload.get("fatigue_level", ""),
                    duration_minutes
                ])
        except Exception as e:
            print(f"[ERROR] Không thể ghi log trip offline: {e}")

    def _log_alert_offline(self, event_type, severity, duration, video_path):
        """Ghi nhận dữ liệu cảnh báo vào file CSV (Bằng chứng offline)"""
        try:
            with open(self.offline_alerts_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.trip_id,
                    self.user_id,
                    event_type,
                    severity,
                    duration,
                    video_path
                ])
        except Exception as e:
            print(f"[ERROR] Không thể ghi log alert offline: {e}")

    def _initialize_session(self):
        """Khởi tạo User Profile và Trip mới để đảm bảo tính toàn vẹn dữ liệu."""
        self.session_start = datetime.now()
        
        # Ghi log offline trước
        self._log_trip_offline("START", "Đang chạy", {})

        if not self.client: return
        try:
            # Tạo bản ghi Trip mới với thời gian bắt đầu hiện tại trên Supabase
            new_trip_payload = {
                "user_id": self.user_id,
                "start_time": self.session_start.isoformat(),
                "status": "Đang chạy"
            }
            res = self.client.table("trips").insert(new_trip_payload).execute()
            
            # Ghi nhận trip_id mới được tạo tự động bởi Supabase
            if res.data:
                self.trip_id = res.data[0]['id']
                print(f"[SUCCESS] Đã khởi tạo phiên làm việc mới trên cloud. Trip ID: {self.trip_id}")
        except Exception as e:
            print(f"[WARNING] Không thể tạo Trip mới tự động trên cloud: {e}. Sử dụng ID mặc định.")

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
        """Đồng bộ dữ liệu thống kê định kỳ lên Supabase và lưu Offline CSV."""
        payload = self._get_metrics_payload(yawns, head_tilts, eye_time, drowsy_count, distracted_count)
        
        # Lưu vào log offline
        self._log_trip_offline("UPDATE", "Đang chạy", payload)

        if not self.client: return
        try:
            self.client.table("trips").update(payload).eq("id", self.trip_id).execute()
            print(f"[*] [SUPABASE] Đồng bộ metrics thành công.")
        except Exception as e:
            print(f"[ERROR] Sync Trip Online: {e}")

    def close_session(self, yawns, head_tilts, eye_time, drowsy_count, distracted_count=0):
        """Kết thúc hành trình: Cập nhật end_time và trạng thái cuối cùng (cả Online/Offline)."""
        if not hasattr(self, 'session_start'): return
        
        end_time = datetime.now()
        duration_minutes = int((end_time - self.session_start).total_seconds() / 60)
        payload = self._get_metrics_payload(yawns, head_tilts, eye_time, drowsy_count, distracted_count)
        
        # Lưu vào log offline
        self._log_trip_offline("CLOSE", "Đã kết thúc", payload, duration_minutes)

        if not self.client: 
            print(f"[*] [OFFLINE] Đã đóng hành trình ({duration_minutes} phút) và lưu vào file log.")
            return
            
        try:
            payload.update({
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
                "status": "Đã kết thúc"
            })
            
            self.client.table("trips").update(payload).eq("id", self.trip_id).execute()
            print(f"[*] [SUPABASE] Đã đóng hành trình ({duration_minutes} phút) trên Cloud.")
        except Exception as e:
            print(f"[ERROR] Close Session Online: {e}")

    def upload_alert(self, event_type, severity, duration, video_path):
        """
        Ghi nhận cảnh báo offline vào máy và tải lên kho lưu trữ Storage của Supabase nếu online.
        """
        # Lưu vào log offline
        self._log_alert_offline(event_type, severity, duration, video_path)

        if not self.client:
            print(f"[*] [OFFLINE] Đã lưu cảnh báo {event_type} cục bộ.")
            return
            
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
            print(f"[ERROR] Upload Alert API Online: {e}")
