"""
=================================================================
SCRIPT HUẤN LUYỆN MÔ HÌNH AI CHO HỆ THỐNG DMS
=================================================================
Mô hình nhận đầu vào là 6 đặc trưng thống kê (features) trích xuất
từ chuỗi thời gian EAR/MAR/HeadPose, KHÔNG PHẢI ảnh.

Đặc trưng đầu vào [6 features]:
  1. ear_mean     - Trung bình EAR (chênh lệch so với baseline)
  2. mar_mean     - Trung bình MAR (chênh lệch so với baseline)  
  3. mar_variance - Biến thiên đạo hàm MAR (phát hiện nói chuyện)
  4. pitch_var    - Biến thiên pitch (gật đầu)
  5. yaw_mean_abs - Trung bình |yaw| (quay đầu ngang)
  6. pitch_raw_mean - Trung bình pitch thô (nhìn lên/xuống)

Nhãn đầu ra [5 classes]:
  0: Normal, 1: Drowsy, 2: Yawning, 3: Talking, 4: Distracted
=================================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "Models", "dms_model_int8.tflite")
NUM_FEATURES = 6       # Số lượng đặc trưng đầu vào
NUM_CLASSES = 5        # Normal, Drowsy, Yawning, Talking, Distracted
EPOCHS = 100           # Số vòng huấn luyện tối đa
BATCH_SIZE = 64        # Kích thước batch

# Tên các nhãn phân loại (phải khớp thứ tự với PredictMaker.py)
LABEL_NAMES = ["Normal", "Drowsy", "Yawning", "Talking", "Distracted"]

# === HẶNG SỐ SCALE FEATURES ===
# Nhân features lên dải giá trị lớn hơn để INT8 quantization không bị mất độ chính xác
# Ví dụ: EAR = -0.12 × 100 = -12.0 (INT8 có thể biểu diễn tốt)
# THỨ TỰ: [ear_mean, mar_mean, mar_var, pitch_var, yaw_abs, pitch_raw_mean]
# GIÁ TRỊ NÀY PHẢI KHớP VỚI PREDICTMAKER.PY!
FEATURE_SCALE = np.array([100.0, 50.0, 500.0, 1.0, 1.0, 1.0], dtype=np.float32)


def generate_synthetic_dataset(num_samples=10000, noise_level=0.15):
    """
    Tạo dữ liệu huấn luyện tổng hợp (Synthetic Data) dựa trên các quy tắc
    Heuristic đã được kiểm chứng trong thực tế.
    
    Mỗi mẫu gồm 6 features + 1 nhãn (label).
    Thêm nhiễu ngẫu nhiên (noise) để mô hình học được vùng chuyển tiếp mượt mà
    giữa các trạng thái, thay vì quyết định cứng nhắc theo ngưỡng.
    
    Tham số:
        num_samples: Tổng số mẫu tạo ra (chia đều cho 5 lớp)
        noise_level: Mức nhiễu Gaussian thêm vào để tăng độ robust
    
    Trả về:
        X: Mảng features shape (N, 6)
        y: Mảng nhãn shape (N,)
    """
    samples_per_class = num_samples // NUM_CLASSES
    # Thêm 20% mẫu khó (hard examples) cho Normal để model phân biệt rõ ranh giới
    extra_normal = samples_per_class // 5
    X_all, y_all = [], []
    rng = np.random.default_rng(seed=42)  # Seed cố định để tái tạo kết quả
    
    for class_idx in range(NUM_CLASSES):
        # --- Khởi tạo features mặc định (trạng thái bình thường) ---
        # [ear_mean, mar_mean, mar_var, pitch_var, yaw_abs, pitch_raw_mean]
        ear_mean = rng.normal(0.0, 0.02, samples_per_class)       # EAR bình thường ~ 0
        mar_mean = rng.normal(0.0, 0.05, samples_per_class)       # MAR bình thường ~ 0
        mar_var = np.abs(rng.normal(0.005, 0.003, samples_per_class))  # Mar variance nhỏ
        pitch_var = np.abs(rng.normal(5.0, 3.0, samples_per_class))    # Pitch ổn định
        yaw_abs = np.abs(rng.normal(5.0, 3.0, samples_per_class))     # Nhìn thẳng
        pitch_raw = rng.normal(0.0, 5.0, samples_per_class)           # Pitch offset nhỏ
        
        if class_idx == 1:
            # === DROWSY: Mắt nhắm (EAR giảm mạnh) ===
            # EAR giảm rõ rệt (trung bình -0.12, một số mẫu đến -0.20)
            ear_mean = rng.normal(-0.12, 0.04, samples_per_class)
            mar_mean = rng.normal(0.0, 0.05, samples_per_class)     # Miệng bình thường
            pitch_var = np.abs(rng.normal(3.0, 2.0, samples_per_class))  # Đầu ít động
            
        elif class_idx == 2:
            # === YAWNING: Miệng há lớn (MAR tăng cao) ===
            mar_mean = rng.normal(0.40, 0.10, samples_per_class)    # MAR tăng mạnh
            mar_var = np.abs(rng.normal(0.01, 0.005, samples_per_class))  # Variance nhẹ (há to đều)
            ear_mean = rng.normal(-0.02, 0.02, samples_per_class)   # Mắt hơi nhắm khi ngáp
            
        elif class_idx == 3:
            # === TALKING: Miệng mở đóng liên tục (MAR variance cao) ===
            mar_mean = rng.normal(0.12, 0.05, samples_per_class)    # MAR hơi cao
            mar_var = np.abs(rng.normal(0.08, 0.025, samples_per_class))  # Variance đạo hàm MAR cao rõ rệt
            
        elif class_idx == 4:
            # === DISTRACTED: Đầu quay lệch (Yaw hoặc Pitch lệch nhiều) ===
            # Tạo 3 loại distracted trộn lẫn
            n_third = samples_per_class // 3
            n_rest = samples_per_class - 2 * n_third
            
            # Loại 1: Quay đầu ngang (yaw lớn)
            yaw_abs[:n_third] = np.abs(rng.normal(55.0, 10.0, n_third))
            # Loại 2: Nhìn xuống/lên quá mức (pitch lệch)
            pitch_raw[n_third:2*n_third] = rng.choice([-1, 1], n_third) * np.abs(rng.normal(40.0, 8.0, n_third))
            # Loại 3: Kết hợp yaw + pitch (nhìn chéo)
            yaw_abs[2*n_third:] = np.abs(rng.normal(35.0, 5.0, n_rest))
            pitch_raw[2*n_third:] = rng.choice([-1, 1], n_rest) * np.abs(rng.normal(25.0, 5.0, n_rest))
        
        # Ghép 6 features thành mảng (N, 6)
        features = np.stack([ear_mean, mar_mean, mar_var, pitch_var, yaw_abs, pitch_raw], axis=1)
        
        # Thêm nhiễu Gaussian toàn cục để tăng tính robust
        noise = rng.normal(0, noise_level, features.shape)
        features += noise * np.array([0.01, 0.02, 0.002, 1.0, 2.0, 2.0])  # Scale nhiễu theo từng feature
        
        X_all.append(features.astype(np.float32))
        y_all.append(np.full(samples_per_class, class_idx, dtype=np.int32))
    
    # --- THÊM HARD NEGATIVE EXAMPLES CHO NORMAL ---
    # Mẫu Normal nhưng ở gần ranh giới (giúp model học NOT drowsy, NOT talking, etc.)
    ear_border = rng.normal(-0.04, 0.01, extra_normal)        # EAR hơi thấp nhưng chưa đủ drowsy
    mar_border = rng.normal(0.15, 0.05, extra_normal)         # MAR hơi cao nhưng chưa yawn
    mvar_border = np.abs(rng.normal(0.02, 0.008, extra_normal))  # mar_var hơi cao nhưng chưa talking
    pvar_border = np.abs(rng.normal(8.0, 3.0, extra_normal))  # Pitch hơi dao động
    yaw_border = np.abs(rng.normal(20.0, 8.0, extra_normal))  # Yaw hơi lệch nhưng OK
    praw_border = rng.normal(0.0, 8.0, extra_normal)          # Pitch raw nhẹ
    border_features = np.stack([ear_border, mar_border, mvar_border, pvar_border, yaw_border, praw_border], axis=1).astype(np.float32)
    X_all.append(border_features)
    y_all.append(np.full(extra_normal, 0, dtype=np.int32))  # Nhãn Normal
    
    # Trộn ngẫu nhiên toàn bộ dataset
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    indices = rng.permutation(len(X))
    
    # Áp dụng FEATURE_SCALE để đưa features về dải giá trị lớn hơn cho INT8
    X = X[indices] * FEATURE_SCALE
    return X, y[indices]


def build_model():
    """
    Xây dựng mạng Dense Network siêu nhẹ cho phân loại 6 features → 5 classes.
    Kiến trúc: 6 → BN → 64 → BN → 32 → 5 (tổng ~3,000 tham số, cực nhẹ cho Jetson Nano).
    BatchNormalization giúp chuẩn hóa dải giá trị features trước khi quantize INT8.
    """
    model = models.Sequential([
        # Chuẩn hóa input - cực kỳ quan trọng cho INT8 quantization
        # EAR/MAR chỉ ~0.1 sẽ được scale lên dải lớn hơn để không bị làm tròn
        layers.BatchNormalization(input_shape=(NUM_FEATURES,)),
        
        # Lớp ẩn 1: Trích xuất đặc trưng phi tuyến từ 6 features đầu vào
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),  # Chuẩn hóa đầu ra giữa các lớp
        layers.Dropout(0.3),  # Chống overfitting
        
        # Lớp ẩn 2: Tinh chỉnh ranh giới phân loại
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Lớp đầu ra: Xác suất cho 5 trạng thái
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def main():
    print("=" * 60)
    print(" HUẤN LUYỆN MÔ HÌNH DMS - KIẾN TRÚC 6-FEATURES")
    print("=" * 60)
    
    # 1. Tạo dữ liệu huấn luyện tổng hợp
    print("\n[INFO] Đang tạo dữ liệu huấn luyện tổng hợp (Synthetic Data)...")
    X, y = generate_synthetic_dataset(num_samples=20000, noise_level=0.15)
    
    # Chia train/validation (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"[INFO] Tổng mẫu: {len(X)} | Train: {len(X_train)} | Validation: {len(X_val)}")
    print(f"[INFO] Phân bố nhãn: {dict(zip(LABEL_NAMES, np.bincount(y)))}")
    
    # 2. Xây dựng model
    print("\n[INFO] Khởi tạo mô hình Dense Network...")
    model = build_model()
    model.summary()
    
    # 3. Huấn luyện với EarlyStopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,                 # Dừng nếu loss không giảm sau 10 epoch
        restore_best_weights=True,   # Giữ trọng số tốt nhất
        verbose=1
    )
    
    print(f"\n[INFO] Bắt đầu huấn luyện {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 4. Đánh giá kết quả trên tập validation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[KẾT QUẢ] Validation Accuracy: {val_acc*100:.1f}% | Loss: {val_loss:.4f}")
    
    # 5. Chuyển đổi sang TFLite INT8
    print("\n[INFO] Đang lượng tử hóa (Quantizing) sang TFLite INT8...")
    
    # Hàm cung cấp dữ liệu đại diện cho lượng tử hóa INT8
    def representative_data_gen():
        """Cung cấp mẫu dữ liệu để TFLite xác định dải giá trị min/max cho INT8."""
        for i in range(min(500, len(X_train))):
            yield [X_train[i:i+1]]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Kích hoạt tối ưu hóa lượng tử hoá toàn phần INT8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ép toàn bộ ops về INT8 - an toàn vì features đã được scale lên dải lớn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # INPUT/OUTPUT giữ float32 để tương thích với PredictMaker.py
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    # 6. Lưu model xuống ổ đĩa
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Hiển thị kích thước file
    model_size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    print(f"\n{'=' * 60}")
    print(f" 🎉 THÀNH CÔNG!")
    print(f" Đã lưu model TFLite tại: {MODEL_SAVE_PATH}")
    print(f" Kích thước: {model_size_kb:.1f} KB")
    print(f" Input shape: (1, {NUM_FEATURES}) float32")
    print(f" Output shape: (1, {NUM_CLASSES}) float32")
    print(f" Labels: {LABEL_NAMES}")
    print(f"{'=' * 60}")
    
    # 7. Kiểm tra nhanh model TFLite vừa tạo
    print("\n[TEST] Chạy inference thử trên model TFLite...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_SAVE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    
    # Test với từng loại trạng thái (GIÁ TRỊ ĐÃ ĐƯỢC SCALE)
    test_cases = {
        "Normal":     np.array([[0.0,   0.0,  0.005, 5.0,  5.0,  0.0]], dtype=np.float32) * FEATURE_SCALE,
        "Drowsy":     np.array([[-0.13, 0.0,  0.005, 3.0,  5.0,  0.0]], dtype=np.float32) * FEATURE_SCALE,
        "Yawning":    np.array([[0.0,   0.40, 0.01,  5.0,  5.0,  0.0]], dtype=np.float32) * FEATURE_SCALE,
        "Talking":    np.array([[0.0,   0.12, 0.08,  5.0,  5.0,  0.0]], dtype=np.float32) * FEATURE_SCALE,
        "Distracted": np.array([[0.0,   0.0,  0.005, 5.0,  55.0, 0.0]], dtype=np.float32) * FEATURE_SCALE,
    }
    
    for expected, features in test_cases.items():
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = LABEL_NAMES[np.argmax(output[0])]
        confidence = np.max(output[0]) * 100
        status = "✅" if predicted == expected else "❌"
        print(f"  {status} Kỳ vọng: {expected:12s} → Dự đoán: {predicted:12s} ({confidence:.0f}%)")


if __name__ == '__main__':
    main()
