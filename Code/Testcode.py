import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pyswarms as ps
import warnings
warnings.filterwarnings('ignore') # Tắt các cảnh báo lặt vặt của Python

# ==========================================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU THẬT
# ==========================================
print("[1/5] Đang đọc và xử lý dữ liệu...")

# TODO: ĐỔI TÊN FILE CSV Ở ĐÂY CHO ĐÚNG VỚI FILE BẠN TẢI VỀ (VD: 'cocomo81.csv')
file_path = 'cocomo81.csv' 

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{file_path}'. Hãy chắc chắn file CSV đang nằm cùng thư mục với script này!")
    exit()

# Xử lý dữ liệu: Xóa các dòng bị thiếu dữ liệu (Bài báo có đề cập tập Desharnais bị thiếu 4 dòng)
df = df.dropna()

# Tách Features (X) và Label (y)
# Giả định cột cuối cùng trong file csv luôn là cột Effort (Chi phí cần dự đoán)
X_data = df.iloc[:, :-1].values 
y_data = df.iloc[:, -1].values

# Tỷ lệ chia 80% Train, 20% Test theo đúng bài báo
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (Scaling)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Lấy số lượng features tự động để nạp vào mạng CNN
n_features = X_train_scaled.shape[1]

# Reshape cho 1D-CNN: (số lượng mẫu, số features, 1)
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], n_features, 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], n_features, 1))

# ==========================================
# 2. HÀM TẠO KIẾN TRÚC MẠNG CNN
# ==========================================
def build_cnn(learning_rate, optimizer_idx):
    model = Sequential([
        Input(shape=(n_features, 1)),
        # Các thông số mạng CNN cấu hình chính xác theo sơ đồ bài báo
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1) # Hồi quy nên Output layer là 1 nơ-ron
    ])
    
    if optimizer_idx == 0:
        opt = Adam(learning_rate=learning_rate)
    elif optimizer_idx == 1:
        opt = SGD(learning_rate=learning_rate)
    else:
        opt = RMSprop(learning_rate=learning_rate)
        
    model.compile(optimizer=opt, loss='mse')
    return model

# ==========================================
# 3. HÀM MỤC TIÊU CHO PSO (FITNESS FUNCTION)
# ==========================================
def cnn_fitness_function(particles):
    n_particles = particles.shape[0]
    losses = np.zeros(n_particles)
    
    for i in range(n_particles):
        lr = particles[i, 0]
        epochs = int(particles[i, 1])
        opt_idx = int(particles[i, 2])
        batch_size = int(particles[i, 3])

        print(f"  Đang huấn luyện hạt {i+1}/{n_particles}...", end="\r")
        
        model = build_cnn(lr, opt_idx)
        # Huấn luyện mô hình
        model.fit(X_train_cnn, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Đánh giá lấy Loss (MSE) trên tập Test để làm thước đo cho PSO
        loss = model.evaluate(X_test_cnn, y_test_scaled, verbose=0)
        losses[i] = loss
        
    return losses

# ==========================================
# 4. CHẠY THUẬT TOÁN PSO
# ==========================================
print(f"[2/5] Bắt đầu khởi chạy PSO với tập dữ liệu chứa {n_features} đặc trưng...")
print("      (Quá trình này có thể mất vài phút tùy vào sức mạnh phần cứng...)")

# Giới hạn chuẩn theo bài báo: lr (0.001-1.0), epochs (1-100), opt (0-2), batch (1-100)
bounds = (np.array([0.001, 1, 0, 1]), np.array([1.0, 100, 2.99, 100]))

options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}

# Cấu hình 50 hạt và 10 vòng lặp theo chuẩn "Experiment settings"
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=4, options=options, bounds=bounds)

cost, pos = optimizer.optimize(cnn_fitness_function, iters=10)

best_lr = pos[0]
best_epochs = int(pos[1])
best_opt_idx = int(pos[2])
best_batch = int(pos[3])
opts = ['Adam', 'SGD', 'RMSprop']

print(f"\n[3/5] HOÀN TẤT TỐI ƯU HÓA! Siêu tham số tốt nhất:")
print(f"      Learning Rate: {best_lr:.4f} | Epochs: {best_epochs} | Optimizer: {opts[best_opt_idx]} | Batch Size: {best_batch}")

# ==========================================
# 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH CUỐI CÙNG
# ==========================================
print("\n[4/5] Đang huấn luyện mô hình cuối cùng dựa trên các tham số tối ưu...")
final_model = build_cnn(best_lr, best_opt_idx)
final_model.fit(X_train_cnn, y_train_scaled, epochs=best_epochs, batch_size=best_batch, verbose=0)

print("[5/5] Đang đánh giá kết quả...\n")
y_pred_scaled = final_model.predict(X_test_cnn, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# Tính toán các chỉ số
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Tính sai số tương đối (tránh lỗi chia cho 0 nếu y_test chứa giá trị 0)
y_test_safe = np.where(y_test == 0, 1e-9, y_test)
mre = np.abs(y_test - y_pred) / y_test_safe
mmre = np.mean(mre)
mdmre = np.median(mre)

# PRED(0.25): Độ chính xác dự báo (sai số <= 25%)
pred_25 = np.mean(mre <= 0.25) * 100

print("="*45)
print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CNN-PSO (CHÍNH THỨC)")
print("="*45)
print(f"MAE:   {mae:.5f}")
print(f"MSE:   {mse:.5f}")
print(f"RMSE:  {rmse:.5f}")
print(f"MMRE:  {mmre:.5f}")
print(f"MdMRE: {mdmre:.5f}")
print(f"PRED:  {pred_25:.2f}%")
print("="*45)