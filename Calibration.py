import numpy as np

class Calibrator:
    def __init__(self, required_frames=300):
        """
        Khởi tạo hệ thống lấy mẫu (Calibration) để tìm ra thông số sinh trắc học
        tiêu chuẩn (Baseline) của tài xế (mắt híp, răng hô...).
        - required_frames: Số lượng khung hình cần thiết (VD: 300 frame tương đương khoảng 10-15s).
        """
        self.required_frames = required_frames
        self.ear_samples = []
        self.mar_samples = []
        self.pitch_raw_samples = []  # Mẫu góc pitch thô để xác định góc nhìn thẳng của tài xế
        self.is_calibrated = False
        self.ear_baseline = 0.0
        self.mar_baseline = 0.0
        self.pitch_raw_baseline = 0.0  # Góc pitch chuẩn (khi nhìn thẳng về phía trước)

        # === NHẬN DIỆN KHUÔN MẶT TÀI XẾ ===
        # Lưu trữ các vector 128 chiều (face encoding) thu thập được trong quá trình calibration
        self.face_encodings = []
        # Vector trung bình đại diện cho khuôn mặt tài xế (được tính sau khi calibration hoàn tất)
        self.driver_encoding = None

    def update(self, ear, mar, pitch_raw=0.0):
        """
        Nạp dữ liệu vào hệ thống chuẩn hoá trong thời gian đầu.
        - pitch_raw: góc pitch thô (độ) từ PnP, dùng để xác định baseline hướng nhìn.
        Trả về True nếu việc hiệu chuẩn (Calibration) hoàn thành.
        """
        if self.is_calibrated:
            return False
            
        self.ear_samples.append(ear)
        self.mar_samples.append(mar)
        self.pitch_raw_samples.append(pitch_raw)  # Ghi nhận góc pitch trong lúc nhìn thẳng
        
        if len(self.ear_samples) >= self.required_frames:
            # Lấy trung bình của toàn bộ các khung hình đã thu thập được
            self.ear_baseline = np.mean(self.ear_samples)
            self.mar_baseline = np.mean(self.mar_samples)
            self.pitch_raw_baseline = np.mean(self.pitch_raw_samples)  # Lưu góc pitch chuẩn
            self.is_calibrated = True

            # Tính vector trung bình đại diện cho khuôn mặt tài xế từ tất cả mẫu đã thu thập
            if len(self.face_encodings) > 0:
                self.driver_encoding = np.mean(self.face_encodings, axis=0)
                print(f"[CALIBRATION] Đã hiệu chuẩn xong: EAR = {self.ear_baseline:.3f}, MAR = {self.mar_baseline:.3f}, Pitch chuẩn = {self.pitch_raw_baseline:.1f}°")
                print(f"[FACE-ID] Đã lưu dấu vân mặt tài xế ({len(self.face_encodings)} mẫu, vector 128-D)")
            else:
                print(f"[CALIBRATION] Đã hiệu chuẩn xong: EAR = {self.ear_baseline:.3f}, MAR = {self.mar_baseline:.3f}, Pitch chuẩn = {self.pitch_raw_baseline:.1f}°")
                print(f"[FACE-ID] CẢNH BÁO: Không thu thập được face encoding nào!")
            return True
            
        return False

    def update_face(self, encoding):
        """
        Thu thập thêm 1 mẫu face encoding (vector 128-D) trong quá trình calibration.
        Chỉ lấy mỗi 5 frame 1 lần để tránh quá nhiều mẫu gần giống nhau (tiết kiệm bộ nhớ).
        """
        if self.is_calibrated:
            return
        # Lấy mẫu mỗi 5 frame để đa dạng hoá góc mặt mà không quá tải bộ nhớ
        if len(self.ear_samples) % 5 == 0:
            self.face_encodings.append(encoding)
        
    def get_progress(self):
        """
        Trả về tiến độ (Tỷ lệ phần trăm 0.0 -> 1.0) của quá trình đo đạc.
        """
        return len(self.ear_samples) / self.required_frames

    def is_driver(self, encoding, threshold=0.6):
        """
        So sánh khuôn mặt đầu vào với khuôn mặt tài xế đã đăng ký.
        - encoding: vector 128-D của khuôn mặt cần kiểm tra
        - threshold: ngưỡng khoảng cách Euclidean (≤ 0.6 = cùng người)
        Trả về True nếu khuôn mặt khớp với tài xế.
        """
        if self.driver_encoding is None:
            return True  # Nếu chưa có face encoding thì mặc định cho qua (fallback)
        distance = np.linalg.norm(self.driver_encoding - encoding)
        return distance <= threshold
        
    def reset(self):
        """
        Khởi động lại toàn bộ chu kỳ học (Clear data cũ).
        GIỮ NGUYÊN driver_encoding vì khuôn mặt tài xế không thay đổi giữa các chu kỳ.
        """
        self.ear_samples.clear()
        self.mar_samples.clear()
        self.pitch_raw_samples.clear()  # Xoá mẫu pitch để tính lại baseline mới
        self.is_calibrated = False
        self.ear_baseline = 0.0
        self.mar_baseline = 0.0
        self.pitch_raw_baseline = 0.0
        # LƯU Ý: KHÔNG reset self.face_encodings và self.driver_encoding
        # vì khuôn mặt tài xế đã được xác định từ lần calibration đầu tiên

