# 🚘 Driver Drowsiness Detection (DMS)
> Hệ thống giám sát tài xế AI thời gian thực - Cảnh báo thông minh, bảo vệ hành trình.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Dlib](https://img.shields.io/badge/Dlib-Recognize-red.svg)](http://dlib.net/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-INT8-orange.svg)](https://www.tensorflow.org/lite)
[![Supabase](https://img.shields.io/badge/Supabase-Sync-green.svg)](https://supabase.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

<div align="center">
  <!-- Vị trí đặt ảnh chụp màn hình dự án thực tế -->
  <img src="assets/IOT18A-Group5.png" alt="DMS Preview" width="1000"/>
</div>

## ✨ Tính Năng Nổi Bật

- **👁️ Giám sát hành vi theo thời gian thực:** Phân tích dữ liệu EAR và MAR để phát hiện chính xác trạng thái nhắm mắt, ngủ gật và ngáp.
- **📵 Cảnh báo mất tập trung:** Dựa trên các góc xoay đầu (Head Pose) để nhận diện khi tài xế không nhìn đường hoặc sử dụng điện thoại.
- **🎯 Bám sát cá nhân hóa:** Tích hợp nhận diện Face ID độc lập qua cơ chế *Calibration* giúp điều chỉnh sai số cá nhân.
- **⚡ Phản hồi siêu tốc với Edge-AI:** Triển khai bằng mô hình mạng Neural lượng tử (*INT8*) 8.7KB kết hợp chiến lược *Frame Skipping*, đảm bảo tối đa hiệu suất trên *Jetson Nano*.
- **☁️ Đồng bộ đám mây lập tức:** Cập nhật báo cáo hành trình định kỳ và tự động upload video bằng chứng của tài xế lên hệ thống Supabase.

## 🛠 Công Nghệ Sử Dụng

- **Ngôn ngữ:** Python 3.8+
- **Computer Vision:** OpenCV, Dlib, Imutils
- **Machine Learning:** TensorFlow Lite
- **Cloud & Storage:** Supabase (PostgreSQL / Buckets)
- **Utilities:** PyGame, gTTS

## 🚀 Bắt Đầu Nhanh

### 1. Cài đặt các thư viện phụ thuộc
Cài đặt trực tiếp qua `pip` trong môi trường ảo của bạn:
```bash
pip install opencv-python numpy dlib pygame gtts supabase python-dotenv imutils scipy tflite-runtime
```

### 2. Chuẩn bị Models & Cấu hình Cloud
- Tải weights cho Dlib (`shape_predictor_68_face_landmarks.dat` và `dlib_face_recognition_resnet_model_v1.dat`) và đặt vào folder `dlib_shape_predictor/`.
- Tạo file `.env` từ `.env.example` và thiết lập kết nối:
```bash
cp .env.example .env
# Chỉnh sửa file .env để điền SUPABASE_URL và SUPABASE_KEY
```

### 3. Vận hành hệ thống
Khởi động hệ thống tại trung tâm giám sát bằng tập lệnh chính:
```bash
python3 DriverDrowsinessDetection.py
```

## 💻 Cách Sử Dụng 

Sau khi chạy phần mềm, quá trình giám sát sẽ tự động bắt đầu:
1. **Giao thức học ban đầu:** Trong vài giây đầu, người dùng giữ mặt hướng thẳng để hệ thống đo đạc baseline chuẩn (*Calibration*).
2. **Kích hoạt tự động:** Mọi hành vi sai phạm nếu diễn ra đủ lâu sẽ báo động cảnh báo với âm thanh và ghi lại đoạn clip 3 giây.

*Ví dụ thay đổi thông số Buffer cho AI Engine (*tại file nhánh `DriverDrowsinessDetection.py`*):*
```python
# Nhỏ số window_size -> Hệ thống sẽ phản hồi nhạy hơn (phù hợp device cấu hình yếu)
decision_maker = DecisionMaker(window_size=15, model_path="Models/dms_model_int8.tflite")
```

## 🛣 Lộ Trình (Roadmap)

- [x] Phát hiện các hành vi tiêu biểu của tài xế (Ngáp, Cúi, Nhắm mắt).
- [x] Train và lượng tử hoá thuật toán INT8.
- [x] Triển khai Serverless Cloud (Supabase).
- [ ] Mở rộng giao diện điều hướng (Web Dashboard cho trung tâm quản trị).
- [ ] Thích ứng Camera Hồng ngoại (IR) sử dụng trong điều kiện thiều sáng vào ban đêm.

## 🤝 Đóng Góp & Giấy Phép

Mọi đóng góp (*Pull requests, Issues*) đều được nhiệt liệt chào đón! Hãy mở Issue đề xuất giải pháp nếu bạn có nhu cầu muốn tạo những tính năng lớn.
Dự án được phân phối theo tiêu chuẩn **MIT License**. Mọi cá nhân đều được quyền chỉnh sửa và tích hợp vào dự án của riêng mình.

---
*Phát triển bởi [IoT-Group 5-K18-IUH] - 2026*
