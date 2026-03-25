import numpy as np
import math
import cv2  # Nhập thư viện xử lý ảnh OpenCV

# Tọa độ 3D chuẩn của các điểm mốc khuôn mặt người (model 3D tham chiếu)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Đầu mũi - landmark 34
    (0.0, -330.0, -65.0),        # Cằm - landmark 9
    (-225.0, 170.0, -135.0),     # Góc trái mắt trái - landmark 37
    (225.0, 170.0, -135.0),      # Góc phải mắt phải - landmark 46
    (-150.0, -150.0, -125.0),    # Góc trái miệng - landmark 49
    (150.0, -150.0, -125.0)      # Góc phải miệng - landmark 55
])

def isRotationMatrix(R):
    """Kiểm tra xem ma trận R có phải là ma trận quay hợp lệ không."""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """
    Chuyển đổi ma trận quay thành góc Euler (Pitch, Yaw, Roll).
    Trả về mảng [x, y, z] tương ứng [pitch, yaw, roll] theo radian.
    """
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])   # Pitch (gật đầu lên/xuống)
        y = math.atan2(-R[2, 0], sy)        # Yaw (quay đầu trái/phải)
        z = math.atan2(R[1, 0], R[0, 0])   # Roll (nghiêng đầu)
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def getHeadTiltAndCoords(size, image_points, frame_height):
    """
    Tính toán góc đầu (Pitch, Yaw, Roll) và tọa độ vẽ đường chỉ hướng.
    
    Trả về:
      - head_tilt_degree: Góc pitch (độ nghiêng dọc), đơn vị độ
      - yaw_degree: Góc yaw (quay ngang trái/phải), đơn vị độ
      - start_point, end_point, end_point_alt: Các điểm để vẽ đường minh họa
    """
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)

    # Ma trận nội tại của camera (giả định không có biến dạng thấu kính)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Không có biến dạng thấu kính

    # Giải bài toán PnP để tính vector quay và tịnh tiến
    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Chiếu điểm 3D (0, 0, 1000) lên ảnh để vẽ đường chỉ hướng mũi
    (nose_end_point2D, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector, translation_vector,
        camera_matrix, dist_coeffs
    )

    # Chuyển rotation vector → ma trận quay → góc Euler
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = rotationMatrixToEulerAngles(rotation_matrix)

    # Tính pitch (độ nghiêng đầu dọc) - giữ nguyên cách tính cũ để tương thích
    head_tilt_degree = abs([-180] - np.rad2deg([euler_angles[0]]))

    # Tính pitch_raw: góc pitch thô có dấu (đơn vị độ)
    # Âm = nhìn lên, Dương = nhìn xuống (so với khi nhìn thẳng)
    pitch_raw_degree = float(np.rad2deg(euler_angles[0]))

    # Tính yaw (góc quay đầu ngang: dương = phải, âm = trái), đơn vị: độ
    yaw_degree = float(np.rad2deg(euler_angles[1]))

    # Tính các điểm vẽ đường minh họa
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    ending_point_alternate = (ending_point[0], frame_height // 2)

    return head_tilt_degree, yaw_degree, pitch_raw_degree, starting_point, ending_point, ending_point_alternate

