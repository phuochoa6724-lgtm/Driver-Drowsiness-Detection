# DRIVER DROWSINESS DETECTION 
> Hệ thống giám sát tài xế AI thời gian thực - Cảnh báo thông minh, bảo vệ hành trình.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Dlib](https://img.shields.io/badge/Dlib-Recognize-red.svg)](http://dlib.net/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-INT8-orange.svg)](https://www.tensorflow.org/lite)
[![Supabase](https://img.shields.io/badge/Supabase-Sync-green.svg)](https://supabase.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

<div align="center">
  <img src="assets/images/IOT18A-Group5.png" alt="DMS Preview" width="1000"/>
</div>

## TÍNH NĂNG NỔI BẬT

- **Giám sát hành vi theo thời gian thực:** Phân tích dữ liệu EAR và MAR để phát hiện chính xác trạng thái nhắm mắt, ngủ gật và ngáp.
- **Cảnh báo mất tập trung:** Dựa trên các góc xoay đầu (Head Pose) để nhận diện khi tài xế không nhìn đường hoặc sử dụng điện thoại.
- **Bám sát cá nhân hóa:** Tích hợp nhận diện Face ID độc lập qua cơ chế *Calibration* giúp điều chỉnh sai số cá nhân.
- **Phản hồi siêu tốc với Edge-AI:** Triển khai bằng mô hình mạng Neural lượng tử (*INT8*) 8.7KB kết hợp chiến lược *Frame Skipping*, đảm bảo tối đa hiệu suất trên *Jetson Nano*.
- **Đồng bộ đám mây & Lưu trữ Offline:** Cập nhật báo cáo hành trình định kỳ và tự động upload video bằng chứng lên hệ thống Supabase. Khi mất kết nối mạng (Offline), hệ thống tự động lưu trữ bằng chứng vào các file `.csv` trong thư mục `logs/`.

## CÔNG NGHỆ SỬ DỤNG

- **Ngôn ngữ:** Python 3.8+
- **Computer Vision:** OpenCV, Dlib, Imutils
- **Machine Learning:** TensorFlow Lite
- **Cloud & Storage:** Supabase (PostgreSQL / Buckets)
- **Utilities:** PyGame, gTTS

## CẤU TRÚC DỰ ÁN

```
Driver-Drowsiness-Detection/
├── main.py                     # Entry point — chạy ứng dụng tại đây
├── src/
│   ├── core/                   # Logic nghiệp vụ cốt lõi
│   │   ├── backend.py          # Kết nối Supabase, đồng bộ dữ liệu
│   │   ├── calibration.py      # Hiệu chuẩn baseline tài xế
│   │   └── alert_handler.py    # Quản lý sự kiện & cảnh báo
│   ├── detection/              # Trích xuất đặc trưng khuôn mặt
│   │   ├── ear.py              # Eye Aspect Ratio
│   │   ├── mar.py              # Mouth Aspect Ratio
│   │   └── head_pose.py        # Góc đầu (Pitch/Yaw/Roll)
│   ├── inference/              # AI inference
│   │   └── predict_maker.py    # TFLite model + Heuristic fallback
│   └── ui/
│       └── ui_helper.py        # Vẽ overlay lên frame Camera
├── models/
│   ├── tflite/                 # dms_model_int8.tflite
│   └── dlib/                   # shape_predictor + face_recognition .dat
├── assets/
│   ├── audio/                  # File âm thanh cảnh báo (.mp3)
│   ├── images/                 # Hình ảnh tĩnh
├── training/
│   ├── train_model.py          # Script huấn luyện model
│   └── data/                   # Dataset (Closed_Eyes, Yawn, head_pose/...)
├── logs/                       # Logs runtime (gitignored)
├── .env.example                # Mẫu biến môi trường
└── temp_alert/                 # Lưu tạm video cảnh báo (gitignored)
```

## BẮT ĐẦU NHANH

###   1. Cài đặt các thư viện phụ thuộc
```bash
pip install opencv-python numpy dlib pygame gtts supabase python-dotenv imutils scipy tflite-runtime
```

###   2. Chuẩn bị Models & Cấu hình Cloud
- Tải weights cho Dlib và đặt vào folder `models/dlib/`:
  - `shape_predictor_68_face_landmarks.dat`
  - `dlib_face_recognition_resnet_model_v1.dat`
- Tạo file `.env`  và thiết lập kết nối:
```bash
# Chỉnh sửa file .env để điền SUPABASE_URL và SUPABASE_KEY
```

##    3. Vận hành hệ thống
```bash
python3 main.py
```

## CÁCH SỬ DỤNG

Sau khi chạy phần mềm, quá trình giám sát sẽ tự động bắt đầu:
1. **Giao thức học ban đầu:** Trong vài giây đầu, người dùng giữ mặt hướng thẳng để hệ thống đo đạc baseline chuẩn (*Calibration*).
2. **Kích hoạt tự động:** Mọi hành vi sai phạm nếu diễn ra đủ lâu sẽ báo động cảnh báo với âm thanh và ghi lại đoạn clip bằng chứng.


## LỘ TRÌNH PHÁT TRIỂN

- [x] Phát hiện các hành vi tiêu biểu của tài xế (Ngáp, Cúi, Nhắm mắt).
- [x] Train và lượng tử hoá thuật toán INT8.
- [x] Triển khai Serverless Cloud (Supabase).
- [x] Tổ chức lại cấu trúc dự án theo chuẩn Python package.
- [ ] Mở rộng giao diện điều hướng (Web Dashboard cho trung tâm quản trị).
- [ ] Thích ứng Camera Hồng ngoại (IR) sử dụng trong điều kiện thiêu sáng vào ban đêm.




*Phát triển bởi [IoT-Group 5-K18-IUH] và dưới sự hướng dẫn của Thầy Phạm Việt Thành- 2026*
