# 🚗 Driver Monitoring System (DMS) - Hệ Thống Giám Sát Tài Xế Thông Minh 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Dlib](https://img.shields.io/badge/dlib-face--recognition-red)](http://dlib.net/)
[![Supabase](https://img.shields.io/badge/Supabase-Database-green)](https://supabase.com/)

Hệ thống giám sát tài xế (DMS) sử dụng trí tuệ nhân tạo (AI) để theo dõi trạng thái của người lái xe trong thời gian thực, nhằm phát hiện sớm các dấu hiệu mệt mỏi, ngủ gật hoặc mất tập trung để đưa ra cảnh báo kịp thời, giúp giảm thiểu rủi ro tai nạn giao thông.

---

## ✨ Tính Năng Nổi Bật (Key Features)

*   **👁️ Phát Hiện Ngủ Gật (Drowsiness):** Theo dõi chỉ số EAR (Eye Aspect Ratio) để nhận diện trạng thái nhắm mắt quá lâu.
*   **🥱 Nhận Diện Ngáp (Yawning):** Phân tích MAR (Mouth Aspect Ratio) để phát hiện hành vi ngáp thường xuyên, là dấu hiệu của sự mệt mỏi.
*   **📵 Giám Sát Mất Tập Trung (Distraction):** Theo dõi tư thế đầu (Head Pose - Yaw, Pitch, Roll) để cảnh báo khi tài xế không nhìn đường hoặc sử dụng điện thoại.
*   **🗣️ Nhận Diện Nói Chuyện (Talking):** Phân loại hành động nói chuyện để phân tích mức độ tập trung.
*   **🔊 Cảnh Báo Âm Thanh Tiếng Việt:** Phát thông báo bằng giọng nói tự nhiên (gTTS/PyGame) khi phát hiện nguy cơ.
*   **🛡️ Nhận Diện Driver Duy Nhất:** Sử dụng Face Recognition (Dlib) để chỉ theo dõi đúng khuôn mặt của tài xế.
*   **🧠 Xử Lý Dự Đoán Trí Tuệ Nhân Tạo (AI-Driven):** Ưu tiên hoàn toàn đầu ra từ mô hình học sâu (TensorFlow) thay vì sử dụng luật cứng để ghi đè kết quả, giúp hệ thống hoạt động ổn định và chính xác hơn trong các tình huống thực tế.
*   **⚖️ Lọc Nhiễu Chống Nhấp Nháy (Anti-Flicker):** Áp dụng Hysteresis để tránh việc AI chập chờn khi tài xế giữ nguyên một trạng thái (Ví dụ: há miệng ngáp lâu không bị ngắt quãng).
*   **☁️ Đồng Bộ Cloud (Supabase):** Tự động tải báo cáo, ảnh chụp bằng chứng và clip video 3 giây.
*   **🛠️ Cá Nhân Hóa (Calibration):** Chế độ hiệu chuẩn tự động trong 100 khung hình đầu tiên.

---

## ⚙️ Cấu Trúc Dự Án (Modular Architecture)

Dự án được thiết kế theo dạng module hóa để dễ dàng bảo trì và mở rộng:

| Module | Chức năng chính |
| :--- | :--- |
| **`DriverDrowsinessDetection.py`** | Luồng thực thi chính (Main Entry), quản lý Camera và tích hợp các module. |
| **`Backend.py`** | Quản lý kết nối **Supabase**. Đồng bộ dữ liệu hành trình và upload bằng chứng (ảnh/video). |
| **`AlertHandler.py`** | Bộ quản lý sự kiện. Theo dõi thời gian, ghi hình MP4 và kích hoạt cảnh báo âm thanh. |
| **`Calibration.py`** | Hiệu chuẩn các chỉ số cá nhân hóa (Baseline) cho từng tài xế. |
| **`PredictMaker.py`** | Core AI: Sử dụng mô hình TFLite (hoặc thuật toán Heuristic fallback) để phân loại trạng thái. |
| **`Features/EAR.py` & `MAR.py`** | Tính toán tỷ lệ mắt (Eye Aspect Ratio) và miệng (Mouth Aspect Ratio). |
| **`Features/HeadPose.py`** | Ước tính các góc quay của đầu (pitch, yaw, roll) để phát hiện nhìn lệch hướng. |
| **`UIHelper.py`** | Hiển thị Overlay thông tin, đồ thị trạng thái AI và đồng hồ lên màn hình. |

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

*   **Ngôn ngữ:** Python 3.
*   **Thư viện Thị giác máy tính:** OpenCV, Dlib.
*   **AI/Inference:** TFLite (TensorFlow Lite) hoặc Heuristic algorithms.
*   **Âm thanh:** Pygame (để phát âm thanh), gTTS (để tạo file audio).
*   **Cơ sở dữ liệu & Lưu trữ:** Cloud Supabase (PostgreSQL & Storage).
*   **Bảo mật:** `python-dotenv` quản lý API Keys.

---

## 🚀 Hướng Dẫn Cài Đặt (Installation)

### 1. Yêu cầu hệ thống
- Linux/MacOS/Windows (Khuyến nghị Linux cho hiệu năng tốt nhất với Dlib).
- Python 3.8 trở lên.
- Camera 720p hoặc cao hơn.

### 2. Cài đặt thư viện
```bash
pip install opencv-python numpy dlib pygame gtts supabase python-dotenv imutils scipy
```
*(Lưu ý: Để ứng dụng chạy được Mô hình AI TFLite thay vì Thuật toán dự phòng, bạn cần cài đặt thêm thư viện TensorFlow hoặc tflite-runtime)*:
- **Trên máy tính (PC):** `pip install tensorflow`
- **Trên thiết bị nhúng (Jetson Nano/Raspberry Pi):** `pip install tflite-runtime`

*(Nếu cài đặt `dlib` gặp lỗi, hãy đảm bảo bạn đã cài `cmake` và `build-essential` trên Linux)*.

### 3. Tải các Model cần thiết
Bạn cần lưu các file sau vào thư mục `dlib_shape_predictor/`:
- `shape_predictor_68_face_landmarks.dat`: Để trích xuất 68 điểm mốc khuôn mặt.
- `dlib_face_recognition_resnet_model_v1.dat`: (Tùy chọn) Để nhận diện khuôn mặt tài xế.

### 4. Cấu hình biến môi trường
Sao chép file mẫu và điền thông tin của bạn:
```bash
cp .env.example .env
```
Mở file `.env` và điền `SUPABASE_URL` và `SUPABASE_KEY` từ trang quản trị dự án Supabase của bạn.

---

## 🎮 Cách Chạy Ứng Dụng (Running)

Mở terminal và chạy lệnh:
```bash
python3 DriverDrowsinessDetection.py
```

**Quá trình hoạt động:**
1.  **Giai đoạn 1 (Calibration - 100 khung hình đầu):** Hệ thống sẽ yêu cầu tài xế nhìn thẳng, mở mắt bình thường để học các chỉ số cơ bản.
2.  **Giai đoạn 2 (Monitoring):** Sau khi hoàn tất hiệu chuẩn, hệ thống bắt đầu giám sát và đưa ra cảnh báo nếu phát hiện trạng thái ngủ gật (>1.5s), ngáp, hoặc mất tập trung.

---

## 📊 Luồng Xử Lý Dữ Liệu (Execution Flow)

1.  **Capture:** Nhận hình ảnh từ camera.
2.  **Verify:** Nhận diện đúng tài xế (nếu được cấu hình).
3.  **Process:** Dùng Dlib lấy 68 landmarks.
4.  **Extract:** Tính toán EAR, MAR, Head Pose.
5.  **Infer:** AI (TFLite/Heuristic) phân loại trạng thái: *Normal, Drowsy, Yawning, Distracted, Talking*.
6.  **Smooth:** Lọc nhiễu trạng thái bằng giải thuật "Majority Voting" qua 15 khung hình.
7.  **Alert:** Nếu trạng thái nguy hiểm kéo dài, kích hoạt cảnh báo âm thanh và ghi hình.
8.  **Sync:** Tải dữ liệu lên Supabase mỗi 60 giây.

---

## 🧠 Huấn Luyện Lại Mô Hình (Training AI)

Dự án có hỗ trợ tự huấn luyện và chuyển đổi (Quantization) siêu nhẹ thành mô hình TFLite (INT8 8-bit) phù hợp chạy trên Jetson Nano.

### Cấu trúc thư mục dữ liệu

Lưu ảnh chuẩn bị huấn luyện vào:
```
Driver-Drowsiness-Detection/
└── TrainAI/
    ├── train/
    │   ├── Open_Eyes/    # Thư mục ảnh mở mắt
    │   └── Closed_Eyes/  # Thư mục ảnh nhắm mắt
    └── train_model.py    # Kịch bản huấn luyện
```

### Cách chạy Script

Yêu cầu cài đặt thêm `tensorflow` (Bạn nên tạo môi trường ảo riêng biệt để huấn luyện nếu cài trên máy tính cá nhân):
```bash
pip install tensorflow
cd TrainAI
python3 train_model.py
```
Sau khi hoàn tất, file TFLite được xuất ra tại `Models/eye_state_int8.tflite` sẵn sàng để nhúng vào hệ thống Edge.

---

## 📁 Dữ Liệu Cảnh Báo (Alert Evidence)
Các bằng chứng bao gồm ảnh chụp màn hình và video clip 3 giây sẽ được lưu tạm tại `temp_alert/` và tự động xóa sau khi đã tải lên thành công lên Supabase Storage.

---

## 📝 Lưu Ý Về Model AI
Dự án được tối ưu hóa cho **Jetson Nano** hoặc các thiết bị cấu hình thấp. Nếu không tìm thấy file `Models/dms_model_int8.tflite`, hệ thống sẽ tự động chuyển sang giải thuật **Heuristic Fallback** dựa trên các ngưỡng sinh học tiêu chuẩn (EAR < 0.2, Yaw > 45°...) mà vẫn đảm bảo độ chính xác cơ bản.

---
*Phát triển bởi [IoT-Group 5-K18-IUH] - 2026*
