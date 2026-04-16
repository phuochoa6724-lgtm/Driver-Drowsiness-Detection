from scipy.spatial import distance as dist
import numpy as np

def mouth_aspect_ratio(mouth):
    """
    Tính tỷ lệ khung miệng (Mouth Aspect Ratio - MAR).
    Dùng để phát hiện ngáp hoặc nói chuyện dựa trên khoảng cách các điểm mốc miệng.
    
    - mouth: mảng tọa độ (x, y) của 20 điểm mốc miệng (landmark 49-68 của dlib).
    - Trả về: giá trị MAR (càng lớn → miệng càng mở rộng)
    """
    # Khoảng cách Euclidean theo chiều dọc (2 cặp điểm: landmark 51-59 và 53-57)
    A = dist.euclidean(mouth[2], mouth[10])  # landmark 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # landmark 53, 57

    # Khoảng cách Euclidean theo chiều ngang (landmark 49-55)
    C = dist.euclidean(mouth[0], mouth[6])   # landmark 49, 55

    # Tính tỷ lệ khung miệng
    mar = (A + B) / (2.0 * C)
    return mar
