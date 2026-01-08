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
N_SAMPLES = 20000       # Số lượng mẫu (20k là đủ tốt)
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
        self.rx_coords = np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel(), np.zeros(self.X_grid.size)))
        self.n_sensors = self.rx_coords.shape[0]
        print(f"Hệ thống khởi tạo: {self.n_sensors} cảm biến (Lưới {grid_size}m)")

    def calculate_los_channel(self, led_pos, rx_pos_list, obj_params=None):
        vec_d = rx_pos_list - led_pos
        dist = np.linalg.norm(vec_d, axis=1)
        dist_sq = dist ** 2
        
        cos_phi = -vec_d[:, 2] / dist
        cos_psi = -vec_d[:, 2] / dist 
        
        H = np.zeros(len(dist))
        
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
            p1 = led_pos[:2]; p2 = rx_pos_list[:, :2]
            d_vec = p2 - p1; f_vec = p1 - np.array([ox, oy])
            a = np.sum(d_vec**2, axis=1)
            b = np.sum(d_vec * f_vec, axis=1) 
            c = np.sum(f_vec**2) - r**2
            delta = b**2 - a*c
            pot_idx = np.where(delta >= 0)
            
            if len(pot_idx[0]) > 0:
                sqrt_delta = np.sqrt(delta[pot_idx])
                t1 = (-b[pot_idx] - sqrt_delta) / a[pot_idx]
                t2 = (-b[pot_idx] + sqrt_delta) / a[pot_idx]
                t_start = np.maximum(0, t1); t_end = np.minimum(1, t2)
                
                valid_intersect = t_start < t_end
                real_idx = pot_idx[0][valid_intersect]
                t_end_real = t_end[valid_intersect]
                z_at_exit = self.H * (1 - t_end_real)
                is_blocked = z_at_exit < h
                H[real_idx[is_blocked]] = 0.0
        return H

# ==============================================================================
# 3. SINH DỮ LIỆU (DATA GENERATION) - LOG INPUT
# ==============================================================================
def generate_training_data(sim, n_samples):
    X = []
    y = []
    print(f"--- Bắt đầu sinh {n_samples} mẫu dữ liệu... ---")
    
    start_time = time.time()
    count = 0
    cos_threshold = np.cos(np.radians(SEMI_ANGLE))
    
    while count < n_samples:
        ox = np.random.uniform(0.5, sim.L - 0.5)
        oy = np.random.uniform(0.5, sim.W - 0.5)
        r = np.random.uniform(0.15, 0.40) 
        h = np.random.uniform(1.2, 1.9)   
        obj = [ox, oy, r, h]
        
        # Check Visibility
        visible_led_count = 0
        obj_top = np.array([ox, oy, h])
        for led in LED_POSITIONS:
            vec = led - obj_top
            if abs(vec[2]) / np.linalg.norm(vec) > cos_threshold:
                 visible_led_count += 1
        
        if visible_led_count < MIN_VISIBLE_SENSORS:
            continue 
            
        rss_features = []
        for led in LED_POSITIONS:
            h_channel = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_channel * LED_POWER
            
            # Add Noise & Log Transform
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            p_rx_noisy = np.maximum(p_rx + noise, 1e-12)
            p_rx_log = 10 * np.log10(p_rx_noisy)
            
            rss_features.append(p_rx_log)
            
        X.append(np.concatenate(rss_features))
        y.append(obj)
        count += 1
        
        if count % 5000 == 0:
            print(f"Đã sinh {count}/{n_samples} mẫu...")
            
    print(f"Hoàn tất. Thời gian: {time.time() - start_time:.2f}s")
    return np.array(X), np.array(y)

# ==============================================================================
# 4. BỘ CÔNG CỤ VISUALIZATION (ĐẦY ĐỦ)
# ==============================================================================

def plot_learning_curve(model, train_rmse, val_rmse, test_rmse):
    """1. Vẽ biểu đồ Loss và so sánh RMSE 3 tập"""
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_curve_, label='Training Loss', color='blue')
    plt.title('Quá trình huấn luyện (Loss Curve)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: So sánh kết quả 3 tập (Bar Chart)
    plt.subplot(1, 2, 2)
    sets = ['Train (80%)', 'Valid (10%)', 'Test (10%)']
    values = [train_rmse, val_rmse, test_rmse]
    bars = plt.bar(sets, values, color=['green', 'orange', 'red'])
    plt.title('So sánh Sai số Vị trí (RMSE)')
    plt.ylabel('RMSE (cm)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.show()

def plot_matlab_style_regression(y_train_true, y_train_pred, 
                                 y_val_true, y_val_pred, 
                                 y_test_true, y_test_pred):
    """2. Vẽ biểu đồ Regression R-Value phong cách MATLAB"""
    print("\n--- Đang vẽ biểu đồ Regression (MATLAB Style)... ---")
    
    y_all_true = np.concatenate([y_train_true, y_val_true, y_test_true])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    
    datasets = [
        (y_train_true, y_train_pred, 'Training'),
        (y_val_true, y_val_pred, 'Validation'),
        (y_test_true, y_test_pred, 'Test'),
        (y_all_true, y_all_pred, 'All')
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    
    for i, (true, pred, title) in enumerate(datasets):
        ax = axs[i]
        t = true.flatten(); o = pred.flatten()
        
        # Tính R và Fit line
        if len(t) > 1:
            R = np.corrcoef(t, o)[0, 1]
            slope, intercept = np.polyfit(t, o, 1)
        else:
            R, slope, intercept = 0, 1, 0
            
        # Vẽ
        ax.scatter(t, o, facecolors='none', edgecolors='k', s=20, label='Data')
        x_vals = np.array([min(t), max(t)])
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color=['blue', 'green', 'red', 'black'][i], linewidth=2, label='Fit')
        ax.plot([min(t), max(t)], [min(t), max(t)], 'k--', alpha=0.5, label='Y = T')
        
        ax.set_title(f'{title}: R={R:.5f}', fontweight='bold')
        ax.set_xlabel('Target'); ax.set_ylabel(f'Output ~= {slope:.2f}*T + {intercept:.2f}')
        ax.legend(loc='upper left'); ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plt.show()

def plot_cdf_error(y_true, y_pred):
    """3. Vẽ CDF phân bố lỗi"""
    errors = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    errors_sorted = np.sort(errors)
    p = 1. * np.arange(len(errors)) / (len(errors) - 1)
    p90_err = np.percentile(errors, 90)
    
    plt.figure(figsize=(8, 6))
    plt.plot(errors_sorted * 100, p, linewidth=2)
    plt.axvline(x=p90_err*100, color='r', linestyle='--', label=f'90% < {p90_err*100:.1f} cm')
    plt.title('CDF - Phân bố lỗi tích lũy (Test Set)')
    plt.xlabel('Lỗi vị trí (cm)'); plt.ylabel('Xác suất')
    plt.legend(); plt.grid(True)
    plt.show()

def visualize_trajectory(sim, model, scaler_X, scaler_y):
    """4. Vẽ quỹ đạo"""
    print("\n--- Đang vẽ quỹ đạo di chuyển... ---")
    theta = np.linspace(0, 2*np.pi, 40)
    path_x = 2.5 + 1.5 * np.cos(theta)
    path_y = 2.5 + 1.5 * np.sin(theta)
    pred_path = []
    
    for i in range(len(path_x)):
        obj = [path_x[i], path_y[i], 0.3, 1.6]
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
    plt.plot(path_x, path_y, 'g--', linewidth=2, label='Thực tế')
    plt.plot(pred_path[:, 0], pred_path[:, 1], 'b-o', markersize=4, label='Dự đoán')
    plt.xlim(0, 5); plt.ylim(0, 5)
    plt.title('Tracking Trajectory'); plt.legend(); plt.grid(True)
    plt.show()

def stress_test(sim, model, scaler_X, scaler_y):
    """5. Stress Test"""
    print("\n--- Đang chạy Stress Test... ---")
    noises = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    rmses = []
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
        print(f"  Noise {n:.0e} -> RMSE: {rmses[-1]:.2f} cm")
        
    plt.figure(figsize=(7, 4))
    plt.plot([str(x) for x in noises], rmses, 'r-o', linewidth=2)
    plt.title('Stress Test: Độ bền với nhiễu')
    plt.xlabel('Mức nhiễu (W)'); plt.ylabel('RMSE (cm)'); plt.grid(True)
    plt.show()

# ==============================================================================
# 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ (MAIN LOGIC)
# ==============================================================================
def train_and_evaluate(X, y, sim):
    print("\n--- Bắt đầu Huấn luyện (Chia tập 80:10:10) ---")
    
    # 1. Chuẩn hóa
    s_X = StandardScaler()
    s_y = StandardScaler()
    X_s = s_X.fit_transform(X)
    y_s = s_y.fit_transform(y)
    
    # 2. CHIA DỮ LIỆU
    X_train, X_temp, y_train, y_temp = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Số lượng mẫu: Train={len(X_train)}, Valid={len(X_val)}, Test={len(X_test)}")
    
    # 3. Model Deep MLP
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )
    
    # 4. Train
    model.fit(X_train, y_train)
    
    # 5. Đánh giá
    def get_metrics(X_in, y_in):
        pred_s = model.predict(X_in)
        pred_real = s_y.inverse_transform(pred_s)
        real = s_y.inverse_transform(y_in)
        dist_err = np.sqrt((real[:,0]-pred_real[:,0])**2 + (real[:,1]-pred_real[:,1])**2)
        return np.mean(dist_err) * 100, real, pred_real

    rmse_train, y_train_true, y_train_pred = get_metrics(X_train, y_train)
    rmse_val, y_val_true, y_val_pred       = get_metrics(X_val, y_val)
    rmse_test, y_test_true, y_test_pred    = get_metrics(X_test, y_test)
    
    print(f"\n>>> KẾT QUẢ RMSE Vị trí:")
    print(f" - Train: {rmse_train:.2f} cm")
    print(f" - Valid: {rmse_val:.2f} cm")
    print(f" - Test : {rmse_test:.2f} cm")
    
    # 6. GỌI VISUALIZATION
    plot_learning_curve(model, rmse_train, rmse_val, rmse_test)
    
    # Vẽ Regression Plot (MATLAB Style)
    plot_matlab_style_regression(y_train_true, y_train_pred, 
                                 y_val_true, y_val_pred, 
                                 y_test_true, y_test_pred)
                                 
    plot_cdf_error(y_test_true, y_test_pred)
    visualize_trajectory(sim, model, s_X, s_y)
    stress_test(sim, model, s_X, s_y)

if __name__ == "__main__":
    sim = VLPSimulator(ROOM_DIM, GRID_SIZE)
    X_data, y_data = generate_training_data(sim, N_SAMPLES)
    train_and_evaluate(X_data, y_data, sim)