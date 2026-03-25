# Driver Drowsiness Detection (Dự án Phát hiện Ngủ gật)

Dưới đây là danh sách các thư viện cần thiết để có thể chạy dự án này thành công. 

## 📦 Danh sách thư viện (Dependencies)

Để chương trình hoạt động bình thường, bạn cần đảm bảo các thư viện sau đã được cài đặt trong môi trường Python của mình (khuyên dùng thông qua `pip`):

1. **`numpy`**: Thư viện xử lý mảng (array) và tính toán đại số tuyến tính.
2. **`scipy`**: Dùng để tính toán khoảng cách Euclidean giữa các điểm ảnh trên mắt và miệng (EAR, MAR).
3. **`opencv-python`** (`cv2`): Thư viện mã nguồn mở dùng cho việc xử lý ảnh, thu thập và hiển thị video từ Camera.
4. **`imutils`**: Thư viện hỗ trợ OpenCV giúp thuận tiện hơn trong các thao tác đọc luồng video, thao tác trên khung hình gốc.
5. **`dlib`**: Cực kỳ quan trọng, được dùng để trích xuất 68 điểm đặc trưng trên khuôn mặt (Facial Landmark - nhận diện mắt, miệng, mũi,...). 
6. **`tensorflow` / `keras`**: Framwork học sâu (Deep Learning), sẽ được sử dụng cho việc load model phục vụ dự án nhận diện (`dms_model_int8.tflite`).
7. **`supabase`**: Thư viện backend-as-a-service để lưu trữ kết quả phân tích hoặc thu thập dữ liệu về máy chủ cơ sở dữ liệu.

## 🚀 Lệnh cài đặt nhanh

Bạn có thể sao chép và dán lệnh sau vào Terminal (hoặc Command Prompt) để cài đặt toàn bộ các thư viện cần thiết bằng lệnh `pip`:

```bash
pip install numpy scipy opencv-python imutils dlib tensorflow supabase
```

> **Lưu ý nhỏ về `dlib`**:
> Để cài đặt thư viện `dlib` thành công (đặc biệt là trên Windows hoặc Linux), thường hệ thống thường yêu cầu phải có trình biên dịch bằng C++ (như Visual Studio C++ Build Tools) và phần mềm **CMake** được cấu hình sẵn trong biến môi trường.

*Bạn cũng có thể copy danh sách ở trên đưa vào file `requirements.txt` để thuận tiện cho việc thiết lập cho các lần sau.*
