from scipy.spatial import distance as dist
import numpy as np

def eye_aspect_ratio(eye):
    """
    Tính tỷ lệ khung mắt (Eye Aspect Ratio - EAR).
    Dùng để phát hiện mắt nhắm/ngủ gật dựa trên khoảng cách các điểm mốc mắt.
    
    - eye: mảng tọa độ (x, y) của 6 điểm mốc mắt từ dlib 68-landmark.
    - Trả về: giá trị EAR (càng nhỏ → mắt càng nhắm, ~0.0 khi nhắm hoàn toàn)
    """
    # Khoảng cách Euclidean theo chiều dọc (2 cặp điểm)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Khoảng cách Euclidean theo chiều ngang (1 cặp điểm)
    C = dist.euclidean(eye[0], eye[3])
    # Tính tỷ lệ khung mắt
    ear = (A + B) / (2.0 * C)
    return ear