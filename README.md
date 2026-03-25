# Driver Monitoring System (DMS) - Phát hiện Ngủ gật & Mất tập trung

Hệ thống giám sát tài xế (DMS) sử dụng trí tuệ nhân tạo (AI) để theo dõi trạng thái của người lái xe trong thời gian thực, nhằm phát hiện sớm các dấu hiệu mệt mỏi, ngủ gật hoặc mất tập trung để đưa ra cảnh báo kịp thời.

## 🛠 Cấu trúc Dự án (Modular Architecture)

Dự án được thiết kế theo dạng module hóa để dễ dàng bảo trì và mở rộng:

*   **`DriverDrowsinessDetection.py`**: File thực thi chính. Quản lý luồng xử lý tổng thể từ Camera đến kết quả cuối cùng.
*   **`Backend.py`**: Quản lý kết nối với **Supabase**. Chịu trách nhiệm đồng bộ dữ liệu hành trình và tải lên (upload) các bằng chứng cảnh báo (ảnh/video).
*   **`AlertHandler.py`**: Bộ quản lý sự kiện. Theo dõi thời gian diễn ra của các trạng thái (Ngủ gật, Ngáp, Mất tập trung), ghi lại video MP4 và kích hoạt tiến trình báo cáo.
*   **`UIHelper.py`**: Chuyên trách hiển thị thông tin lên màn hình (Overlay), bao gồm trạng thái AI, đồng hồ, và các thông số kỹ thuật.
*   **Các Module Tính năng**:
    *   `Calibration.py`: Hiệu chuẩn các chỉ số cá nhân hóa cho từng driver.
    *   `PredictMaker.py`: Chạy mô hình TFLite để dự đoán trạng thái dựa trên dữ liệu chuỗi thời gian.
    *   `EAR.py`, `MAR.py`: Tính toán tỷ lệ mắt (Eye Aspect Ratio) và miệng (Mouth Aspect Ratio).
    *   `HeadPose.py`: Ước tính tư thế đầu (pitch, yaw, roll).

## ⚙️ Luồng hoạt động của hệ thống (Execution Flow)

Vòng lặp chính trong `DriverDrowsinessDetection.py` hoạt động theo 8 bước:

1.  **Đọc Frame**: Thu nhận hình ảnh từ Camera và tiền xử lý (resize, grayscale).
2.  **Nhận diện Driver**: Sử dụng dlib để tìm khuôn mặt và mã hóa (face encoding) để đảm bảo chỉ theo dõi người lái.
3.  **Xử lý Mất khuôn mặt**: Nếu không thấy mặt trong một khoảng thời gian, hệ thống ghi nhận trạng thái **Distracted**.
4.  **Trích xuất Đặc trưng**: Tính toán các chỉ số EAR, MAR và Head Pose từ 68 điểm mốc (landmarks).
5.  **Hiệu chuẩn (Calibration)**: Trong 100 frame đầu, hệ thống "học" các chỉ số cơ bản của tài xế khi ở trạng thái bình thường.
6.  **Dự đoán trạng thái (AI Prediction)**: Sau khi hiệu chuẩn, AI sẽ phân loại trạng thái: *Normal, Drowsy, Yawning, Distracted, Talking*.
7.  **Quản lý Cảnh báo**: Nếu trạng thái bất thường kéo dài vượt ngưỡng (ví dụ: ngủ gật > 1.5s), hệ thống sẽ tự động chụp ảnh và quay video bằng chứng.
8.  **Đồng bộ Backend**: Gửi dữ liệu thống kê tổng hợp lên hệ thống quản lý mỗi 60 giây.

## 🚀 Hướng dấn sử dụng

1.  **Cài đặt thư viện**: Xem chi tiết danh sách và lệnh cài đặt tại [library.md](file:///home/nph/Downloads/Driver-Drowsiness-Detection/library.md).
2.  **Cấu hình**: Cung cấp các biến môi trường `SUPABASE_URL`, `SUPABASE_KEY` và các ID cần thiết trong file `.env`.
3.  **Chạy dự án**:
    ```bash
    python3 DriverDrowsinessDetection.py
    ```

## 🎥 Dữ liệu Cảnh báo
Các ảnh và video được tạo ra trong quá trình cảnh báo sẽ được lưu tạm thời tại thư mục `temp_alert/` trước khi được tải lên cloud.
