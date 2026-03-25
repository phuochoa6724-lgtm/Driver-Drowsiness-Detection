import numpy as np
from collections import deque
import cv2 # Nhập thư viện xử lý ảnh OpenCV (Computer Vision)
import os

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("[CẢNH BÁO] Không tìm thấy tflite_runtime. Sẽ sử dụng Thuật toán thay thế (Heuristic Fallback) để mô phỏng suy luận AI.")

class DecisionMaker:
    def __init__(self, window_size=60, model_path="Models/dms_model_int8.tflite"):
        """
        Khởi tạo module Ra quyết định với bộ đệm (Buffer) và suy luận AI đa tầng.
        - window_size: số lượng khung hình (frame) lưu trong cửa sổ trượt (trung bình 1-2 giây).
        - model_path: Đường dẫn TFLite.
        """
        self.window_size = window_size
        
        # Khởi tạo các hàng đợi hai đầu (deque) với kích thước cố định.
        self.ear_buffer = deque(maxlen=window_size)
        self.mar_buffer = deque(maxlen=window_size)
        self.pitch_buffer = deque(maxlen=window_size)
        self.yaw_buffer = deque(maxlen=window_size)      # Góc quay ngang đầu (trái/phải)
        self.pitch_raw_buffer = deque(maxlen=window_size)  # Góc pitch thô có dấu (để phân biệt nhìn lên/xuống)
        
        # BỘ LỌC LÀM MƯỢT (Smoothing): Lưu lịch sử 15 trạng thái gần nhất để lấy số đông
        self.smoothing_window = 15
        self.state_history = deque(maxlen=self.smoothing_window)
        
        # Danh sách các nhãn phân loại đầu ra của hệ thống
        self.labels = ["Normal", "Drowsy", "Yawning", "Talking", "Distracted"]
        
        self.interpreter = None
        # Thiết lập TFLite Interpreter nếu có tflite và file tồn tại
        if TFLITE_AVAILABLE and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                print(f"[LỖI TFLITE] {e}. Chuyển sang Fallback.")
                self.interpreter = None

    def update_buffer(self, ear, mar, pitch, yaw=0.0, pitch_raw=0.0,
                      ear_baseline=0.0, mar_baseline=0.0, pitch_raw_baseline=0.0):
        """
        Cập nhật dữ liệu vào Buffer trượt theo thời gian.
        - pitch: góc phái sinh từ head_tilt_degree (cũ)
        - yaw: góc quay ngang trái/phải (độ)
        - pitch_raw: góc pitch thô có dấu (độ), âm = nhìn lên, dương = nhìn xuống
        - pitch_raw_baseline: giá trị pitch_raw khi nhìn thẳng (từ calibration)
        """
        self.ear_buffer.append(ear - ear_baseline)
        self.mar_buffer.append(mar - mar_baseline)
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        # Lưu độ lệch pitch so với baseline (dương = nhìn xuống, âm = nhìn lên)
        self.pitch_raw_buffer.append(pitch_raw - pitch_raw_baseline)

    def extract_features(self):
        """
        Trích xuất đặc trưng (Features) từ chuỗi thời gian (time-series).
        """
        # Trả về rỗng nếu chưa thu thập đủ chuỗi thời gian 
        if len(self.ear_buffer) < self.window_size:
            return None 
        
        ear_array = np.array(self.ear_buffer)
        mar_array = np.array(self.mar_buffer)
        pitch_array = np.array(self.pitch_buffer)
        yaw_array = np.array(self.yaw_buffer)
        pitch_raw_array = np.array(self.pitch_raw_buffer)
        
        ear_mean = np.mean(ear_array)
        mar_mean = np.mean(mar_array)
        
        # Đạo hàm của chuyển động miệng (phát hiện nói chuyện)
        mar_grad = np.gradient(mar_array)
        mar_variance = np.var(mar_grad) 
        
        pitch_variance = np.var(pitch_array)              # Biến thiên pitch (gật đầu)
        yaw_mean_abs = np.mean(np.abs(yaw_array))         # Trung bình độ lớn yaw (nhìn ngang)
        pitch_raw_mean = np.mean(pitch_raw_array)         # Chênh lệch pitch trung bình (+ xuống, - lên)
        
        # Đặc trưng thống kê [6 features]
        features = np.array([[ear_mean, mar_mean, mar_variance, pitch_variance, yaw_mean_abs, pitch_raw_mean]], dtype=np.float32)
        return features

    def _heuristic_fallback(self, features):
        """
        Dự đoán nhãn dựa trên quy tắc toán học (Heuristic) khi không có Model TFLite.
        Sử dụng ngưỡng theo tiêu chuẩn Cảnh sát giao thông Việt Nam.
        """
        ear_mean, mar_mean, mar_var, pitch_var, yaw_abs, pitch_raw_mean = features[0]
        
        # Ưu tiên 1: Mắt nhắm → ngủ gật (nguy hiểm nhất)
        if ear_mean < -0.06:
            return "Drowsy"
        
        # Ưu tiên 2: Kiểm tra mất tập trung theo ngưỡng CSGT
        # Yaw > 45° (quay ngang) → không nhìn đường
        if yaw_abs > 45.0:
            return "Distracted"
        # Nhìn lên quá 40° hoặc xuống quá 30° → không nhìn đường
        if pitch_raw_mean < -40.0 or pitch_raw_mean > 30.0:
            return "Distracted"
        # Góc chéo: yaw > 30° KẼT HỢP pitch ngang trục > 20° → nhìn chéo sang cạnh
        if yaw_abs > 30.0 and (pitch_raw_mean < -20.0 or pitch_raw_mean > 15.0):
            return "Distracted"
        
        # Ưu tiên 3: Ngáp
        if mar_mean > 0.25:
            return "Yawning"
        # Ưu tiên 4: Nói chuyện
        if mar_var > 0.03:
            return "Talking"
        # Ưu tiên 5: Mất tập trung do biến thiên pitch
        if pitch_var > 30.0:
            return "Distracted"
            
        return "Normal"

    def predict_state(self):
        """
        Tiến hành chạy mô hình dự đoán (Inference).
        """
        features = self.extract_features()
        if features is None:
            return "Normal"  
        
        state = "Normal"
        # 1. Gọi mạng Neural TFLite nếu có sẵn
        if self.interpreter is not None:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], features)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
                predicted_idx = np.argmax(predictions[0])
                state = self.labels[predicted_idx]
            except Exception as e:
                state = self._heuristic_fallback(features)
        else:
            # 2. Hoặc mô phỏng AI bằng quy tắc tĩnh toán học
            state = self._heuristic_fallback(features)
        
        # 3. Luật cứng theo tiêu chuẩn CSGT (ưu tiên tuyệt đối, ghi đè kết quả model)
        current_pitch = self.pitch_buffer[-1]
        current_ear_delta = self.ear_buffer[-1]
        current_yaw = self.yaw_buffer[-1] if self.yaw_buffer else 0.0
        current_pitch_raw = self.pitch_raw_buffer[-1] if self.pitch_raw_buffer else 0.0
        
        # Quy tắc 1 (cao nhất): mắt sụp + gục đầu → Drowsy
        if current_ear_delta < -0.1 and current_pitch > 20.0:
            state = "Drowsy"
        # Quy tắc 2: Quay ngang > 45° (vượt ngưỡng nhìn gương chiếu hậu) → Distracted
        elif abs(current_yaw) > 45.0:
            state = "Distracted"
        # Quy tắc 3: Nhìn lên quá 40° hoặc xuống quá 30° → Distracted
        elif current_pitch_raw < -40.0 or current_pitch_raw > 30.0:
            state = "Distracted"
        # Quy tắc 4: Góc chéo (nhìn chéo về phía không phải đường) → Distracted
        elif abs(current_yaw) > 30.0 and (current_pitch_raw < -20.0 or current_pitch_raw > 15.0):
            state = "Distracted"
            
        # 4. BỘ LỌC LÀM MƯỢT (Smoothing): Majority Voting
        # Lưu trạng thái thô vào lịch sử
        self.state_history.append(state)
        
        # Lấy trạng thái xuất hiện nhiều nhất trong smoothing_window (15 frames)
        if len(self.state_history) > 0:
            most_common_state = max(set(self.state_history), key=list(self.state_history).count)
            return most_common_state
            
        return state
