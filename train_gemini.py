import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIGURATION)
# ==============================================================================

# Thông số phòng
ROOM_DIM = [5, 5, 3]    # Dài, Rộng, Cao (m)
GRID_SIZE = 0.2         # Kích thước lưới (m)
N_SAMPLES = 20000       # Số lượng mẫu (20k là đủ tốt cho MLP)
MIN_VISIBLE_SENSORS = 1 # Ít nhất 1 đèn nhìn thấy vật thể

# Thông số Đèn LED (Transmitter)
LED_POWER = 10.0        # Watts
SEMI_ANGLE = 60         # Độ
LAMBERTIAN_ORDER = -np.log(2) / np.log(np.cos(np.radians(SEMI_ANGLE)))

# Thông số Bộ thu (Receiver - PD)
PD_AREA = 1e-4          # 1 cm^2
FOV = 60                # Độ
REFRACTIVE_INDEX = 1.5
FILTER_GAIN = 1.0
CONC_GAIN = (REFRACTIVE_INDEX**2) / (np.sin(np.radians(FOV))**2)

# Vị trí 4 đèn LED (Treo trên trần 3m)
LED_POSITIONS = np.array([
    [1.25, 1.25, 3.0], 
    [3.75, 1.25, 3.0],
    [3.75, 3.75, 3.0], 
    [1.25, 3.75, 3.0]
])

# ==============================================================================
# 2. LỚP MÔ PHỎNG VẬT LÝ (PHYSICS ENGINE)
# ==============================================================================
class VLPSimulator:
    def __init__(self, room_dim, grid_size):
        self.L, self.W, self.H = room_dim
        self.grid_size = grid_size
        
        # Tạo lưới cảm biến trên sàn
        x = np.arange(0, self.L + grid_size/100, grid_size)
        y = np.arange(0, self.W + grid_size/100, grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        
        # Danh sách tọa độ các cảm biến (N_sensors x 3)
        self.rx_coords = np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel(), np.zeros(self.X_grid.size)))
        self.n_sensors = self.rx_coords.shape[0]
        print(f"Hệ thống khởi tạo: {self.n_sensors} cảm biến (Lưới {grid_size}m)")

    def calculate_los_channel(self, led_pos, rx_pos_list, obj_params=None):
        """Tính toán kênh truyền LoS có xét đến vật cản (Shadowing)"""
        vec_d = rx_pos_list - led_pos
        dist = np.linalg.norm(vec_d, axis=1)
        dist_sq = dist ** 2
        
        # Cosine góc phát (Phi) và góc thu (Psi)
        # Vector pháp tuyến đèn hướng xuống (0,0,-1), Vector pháp tuyến PD hướng lên (0,0,1)
        cos_phi = -vec_d[:, 2] / dist
        cos_psi = -vec_d[:, 2] / dist 
        
        H = np.zeros(len(dist))
        
        # Kiểm tra FOV
        fov_rad = np.radians(FOV)
        valid_indices = (cos_psi >= np.cos(fov_rad)) & (cos_phi > 0)
        
        const_factor = ((LAMBERTIAN_ORDER + 1) * PD_AREA) / (2 * np.pi)
        
        H[valid_indices] = (const_factor / dist_sq[valid_indices]) * \
                           (cos_phi[valid_indices] ** LAMBERTIAN_ORDER) * \
                           cos_psi[valid_indices] * \
                           FILTER_GAIN * CONC_GAIN

        # Xử lý Bóng râm (Blockage Logic)
        if obj_params is not None:
            ox, oy, r, h = obj_params
            
            # Chiếu xuống 2D để tìm giao điểm đường tròn
            p1 = led_pos[:2]
            p2 = rx_pos_list[:, :2]
            d_vec = p2 - p1
            f_vec = p1 - np.array([ox, oy])
            
            a = np.sum(d_vec**2, axis=1)
            b = np.sum(d_vec * f_vec, axis=1) 
            c = np.sum(f_vec**2) - r**2
            
            delta = b**2 - a*c
            pot_idx = np.where(delta >= 0)
            
            if len(pot_idx[0]) > 0:
                sqrt_delta = np.sqrt(delta[pot_idx])
                t1 = (-b[pot_idx] - sqrt_delta) / a[pot_idx]
                t2 = (-b[pot_idx] + sqrt_delta) / a[pot_idx]
                
                # Tìm đoạn giao cắt thực sự trên đoạn thẳng nối LED-PD
                t_start = np.maximum(0, t1)
                t_end = np.minimum(1, t2)
                
                valid_intersect = t_start < t_end
                
                real_idx = pot_idx[0][valid_intersect]
                t_end_real = t_end[valid_intersect]
                
                # Kiểm tra độ cao tại điểm thoát ra khỏi vùng trụ
                z_at_exit = self.H * (1 - t_end_real)
                
                # Nếu tia sáng đi thấp hơn chiều cao vật thể -> Bị chặn
                is_blocked = z_at_exit < h
                
                # Gán H=0 cho các tia bị chặn
                blocked_indices = real_idx[is_blocked]
                H[blocked_indices] = 0.0
                
        return H

# ==============================================================================
# 3. SINH DỮ LIỆU (DATA GENERATION) - ĐÃ CẢI TIẾN
# ==============================================================================
def generate_training_data(sim, n_samples):
    X = []
    y = []
    print(f"--- Bắt đầu sinh {n_samples} mẫu dữ liệu (Input: Log-Scale) ---")
    
    start_time = time.time()
    count = 0
    cos_threshold = np.cos(np.radians(SEMI_ANGLE))
    
    while count < n_samples:
        # 1. Random vị trí và kích thước vật thể
        ox = np.random.uniform(0.5, sim.L - 0.5)
        oy = np.random.uniform(0.5, sim.W - 0.5)
        r = np.random.uniform(0.15, 0.40) 
        h = np.random.uniform(1.2, 1.9)   
        obj = [ox, oy, r, h]
        
        # 2. Kiểm tra điều kiện nhìn thấy (Visibility Check) - Đã sửa lỗi vector
        visible_led_count = 0
        obj_top_center = np.array([ox, oy, h])
        
        for led in LED_POSITIONS:
            # Vector từ Vật thể lên LED
            vec_obj_to_led = led - obj_top_center
            dist = np.linalg.norm(vec_obj_to_led)
            # Góc phát so với trục thẳng đứng của đèn
            # Cos(theta) = dot(vec_down, vec_led_to_obj) / dist
            # Vec từ đèn xuống vật thể = -vec_obj_to_led. Z component sẽ âm.
            # Ta lấy độ lớn Z chia cho dist
            if abs(vec_obj_to_led[2]) / dist > cos_threshold:
                 visible_led_count += 1
        
        if visible_led_count < MIN_VISIBLE_SENSORS:
            continue 
            
        # 3. Tính toán RSS và Log-Transform
        rss_features = []
        for led in LED_POSITIONS:
            h_channel = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_channel * LED_POWER
            
            # Thêm nhiễu nền (Noise Floor)
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            p_rx_noisy = np.maximum(p_rx + noise, 1e-12) # Tránh log(0)
            
            # --- CẢI TIẾN QUAN TRỌNG: CHUYỂN SANG DB (LOG) ---
            # Giúp mạng Neural học tốt hơn nhiều so với Linear Watt
            p_rx_log = 10 * np.log10(p_rx_noisy)
            
            rss_features.append(p_rx_log)
            
        X_sample = np.concatenate(rss_features)
        
        X.append(X_sample)
        y.append(obj)
        count += 1
        
        if count % 2000 == 0:
            print(f"Đã sinh {count}/{n_samples} mẫu...")
            
    print(f"Hoàn tất sinh dữ liệu. Thời gian: {time.time() - start_time:.2f}s")
    return np.array(X), np.array(y)

# ==============================================================================
# 4. BỘ CÔNG CỤ VISUALIZATION (BIỂU ĐỒ)
# ==============================================================================
def plot_learning_curve(model):
    """Vẽ Loss và R2 Score"""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_curve_, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.grid(True); plt.legend()
    
    # Validation
    if hasattr(model, 'validation_scores_'):
        plt.subplot(1, 2, 2)
        plt.plot(model.validation_scores_, label='Val Score', color='orange')
        plt.title('Validation Score (R2)')
        plt.xlabel('Epochs'); plt.ylabel('R2 Score')
        plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

def plot_cdf_error(y_true, y_pred):
    """Vẽ CDF sai số định vị"""
    errors = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    errors_sorted = np.sort(errors)
    p = 1. * np.arange(len(errors)) / (len(errors) - 1)
    
    mean_err = np.mean(errors)
    p90_err = np.percentile(errors, 90)
    
    plt.figure(figsize=(8, 6))
    plt.plot(errors_sorted * 100, p, linewidth=2, label='Position Error')
    plt.axvline(x=p90_err*100, color='r', linestyle='--', label=f'90% < {p90_err*100:.1f}cm')
    plt.axvline(x=mean_err*100, color='g', linestyle='--', label=f'Mean = {mean_err*100:.1f}cm')
    plt.title('CDF of Positioning Error')
    plt.xlabel('Error (cm)'); plt.ylabel('Probability')
    plt.grid(True); plt.legend()
    plt.show()

def visualize_trajectory(sim, model, scaler_X, scaler_y):
    """Mô phỏng quỹ đạo di chuyển"""
    print("\n--- Đang vẽ quỹ đạo (Trajectory)... ---")
    theta = np.linspace(0, 2*np.pi, 40)
    path_x = 2.5 + 1.5 * np.cos(theta)
    path_y = 2.5 + 1.5 * np.sin(theta)
    
    pred_path = []
    
    for i in range(len(path_x)):
        obj = [path_x[i], path_y[i], 0.3, 1.6] # Vật thể test cố định R, H
        
        rss_features = []
        for led in LED_POSITIONS:
            h_val = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_val * LED_POWER
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            p_rx_log = 10 * np.log10(np.maximum(p_rx + noise, 1e-12))
            rss_features.append(p_rx_log)
            
        X_in = scaler_X.transform(np.concatenate(rss_features).reshape(1, -1))
        y_out = scaler_y.inverse_transform(model.predict(X_in))
        pred_path.append(y_out[0, :2])
        
    pred_path = np.array(pred_path)
    
    plt.figure(figsize=(6, 6))
    plt.plot(path_x, path_y, 'g--', linewidth=2, label='Ground Truth')
    plt.plot(pred_path[:, 0], pred_path[:, 1], 'b-o', markersize=4, label='Predicted')
    plt.xlim(0, 5); plt.ylim(0, 5)
    plt.title('Circular Trajectory Tracking')
    plt.legend(); plt.grid(True)
    plt.show()

def stress_test(sim, model, scaler_X, scaler_y):
    """Kiểm tra độ bền với nhiễu"""
    print("\n--- Chạy Stress Test... ---")
    noises = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    rmses = []
    
    # Tạo tập test cố định
    test_objs = [[np.random.uniform(1,4), np.random.uniform(1,4), 0.3, 1.5] for _ in range(50)]
    
    for n in noises:
        errs = []
        for obj in test_objs:
            rss_f = []
            for led in LED_POSITIONS:
                h_v = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
                p_r = np.maximum(h_v * LED_POWER + np.random.normal(0, n, len(h_v)), 1e-12)
                rss_f.append(10 * np.log10(p_r))
            
            X_in = scaler_X.transform(np.concatenate(rss_f).reshape(1, -1))
            pred = scaler_y.inverse_transform(model.predict(X_in))
            errs.append(np.sqrt((obj[0]-pred[0,0])**2 + (obj[1]-pred[0,1])**2))
            
        rmses.append(np.mean(errs)*100)
        print(f"Noise {n:.0e} -> RMSE: {rmses[-1]:.2f} cm")
        
    plt.figure(figsize=(6, 4))
    plt.plot([str(x) for x in noises], rmses, 'r-o')
    plt.xlabel('Noise Level (W)'); plt.ylabel('RMSE (cm)')
    plt.title('Robustness Test')
    plt.grid(True); plt.show()

# ==============================================================================
# 5. HUẤN LUYỆN MODEL (MAIN TRAINING)
# ==============================================================================
def train_and_evaluate(X, y, sim):
    print("\n--- Training Deep MLP Regressor ---")
    
    # Chuẩn hóa
    s_X = StandardScaler()
    s_y = StandardScaler()
    X_s = s_X.fit_transform(X)
    y_s = s_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.1, random_state=42)
    
    # Khởi tạo Model (Cấu trúc sâu hơn để học tốt feature phức tạp)
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Đánh giá cơ bản
    preds = s_y.inverse_transform(model.predict(X_test))
    real = s_y.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(real, preds, multioutput='raw_values'))
    
    print(f"\n>>> KẾT QUẢ CUỐI CÙNG (RMSE):")
    print(f"Vị trí X: {rmse[0]*100:.2f} cm")
    print(f"Vị trí Y: {rmse[1]*100:.2f} cm")
    print(f"Bán kính: {rmse[2]*100:.2f} cm")
    print(f"Chiều cao: {rmse[3]*100:.2f} cm")
    
    # Gọi các biểu đồ
    plot_learning_curve(model)
    plot_cdf_error(real, preds)
    visualize_trajectory(sim, model, s_X, s_y)
    stress_test(sim, model, s_X, s_y)

# ==============================================================================
# MAIN PROGRAM
# ==============================================================================
if __name__ == "__main__":
    # 1. Khởi tạo môi trường
    sim = VLPSimulator(ROOM_DIM, GRID_SIZE)
    
    # 2. Sinh dữ liệu (Có thể mất 1-2 phút)
    X_data, y_data = generate_training_data(sim, N_SAMPLES)
    
    # 3. Train & Visualize
    train_and_evaluate(X_data, y_data, sim)