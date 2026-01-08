import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

# --- CẤU HÌNH ---
# Tăng mẫu lên mức tối thiểu để hội tụ
N_SAMPLES = 20000       
# Các tham số khác giữ nguyên
ROOM_DIM = [5, 5, 3]
GRID_SIZE = 0.2
MIN_VISIBLE_SENSORS = 1 
LED_POWER = 10.0
SEMI_ANGLE = 60
LAMBERTIAN_ORDER = -np.log(2) / np.log(np.cos(np.radians(SEMI_ANGLE)))
PD_AREA = 1e-4
FOV = 60
REFRACTIVE_INDEX = 1.5
FILTER_GAIN = 1.0
CONC_GAIN = (REFRACTIVE_INDEX**2) / (np.sin(np.radians(FOV))**2)

LED_POSITIONS = np.array([
    [1.25, 1.25, 3.0], [3.75, 1.25, 3.0],
    [3.75, 3.75, 3.0], [1.25, 3.75, 3.0]
])

# ... (Giữ nguyên class VLPSimulator như cũ) ...
class VLPSimulator:
    def __init__(self, room_dim, grid_size):
        self.L, self.W, self.H = room_dim
        self.grid_size = grid_size
        x = np.arange(0, self.L + grid_size/100, grid_size)
        y = np.arange(0, self.W + grid_size/100, grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.rx_coords = np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel(), np.zeros(self.X_grid.size)))
        self.n_sensors = self.rx_coords.shape[0]

    def calculate_los_channel(self, led_pos, rx_pos_list, obj_params=None):
        # ... (Copy lại y nguyên logic hàm calculate_los_channel của bạn ở đây) ...
        # (Để tiết kiệm không gian tôi không paste lại, hãy giữ nguyên logic đã sửa ở bước trước)
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
                           cos_psi[valid_indices] * FILTER_GAIN * CONC_GAIN
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
                t_start = np.maximum(0, t1)
                t_end = np.minimum(1, t2)
                valid_intersect = t_start < t_end
                real_idx = pot_idx[0][valid_intersect]
                t_end_real = t_end[valid_intersect]
                z_at_exit = self.H * (1 - t_end_real)
                is_blocked = z_at_exit < h
                H[real_idx[is_blocked]] = 0.0
        return H

# --- CẢI TIẾN: INPUT LOGARIT ---
def generate_training_data(sim, n_samples):
    X = []
    y = []
    print(f"--- Đang sinh {n_samples} mẫu (Chế độ Log-Input) ---")
    
    start = time.time()
    count = 0
    cos_threshold = np.cos(np.radians(SEMI_ANGLE))
    
    while count < n_samples:
        ox = np.random.uniform(0.5, sim.L - 0.5)
        oy = np.random.uniform(0.5, sim.W - 0.5)
        r = np.random.uniform(0.15, 0.40) 
        h = np.random.uniform(1.2, 1.9)   
        obj = [ox, oy, r, h]
        
        # Check Visible (Giữ logic đã fix)
        visible_count = 0
        obj_top = np.array([ox, oy, h])
        for led in LED_POSITIONS:
            vec = obj_top - led
            if -vec[2]/np.linalg.norm(vec) > cos_threshold: visible_count += 1
        
        if visible_count < MIN_VISIBLE_SENSORS: continue

        # Tính RSS
        rss_features = []
        for led in LED_POSITIONS:
            h_val = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_val * LED_POWER
            
            # Thêm nhiễu nền thực tế hơn (Thermal noise + Shot noise)
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            p_rx_noisy = np.maximum(p_rx + noise, 1e-12) # Tránh log(0) hoặc log(am)
            
            # --- CẢI TIẾN QUAN TRỌNG: CHUYỂN SANG LOG (dB) ---
            # Input đầu vào dạng tuyến tính (Watts) rất khó học.
            # Chuyển sang dB giúp phân phối dữ liệu đều hơn.
            p_rx_log = 10 * np.log10(p_rx_noisy)
            
            rss_features.append(p_rx_log)

        X.append(np.concatenate(rss_features))
        y.append(obj)
        count += 1
        if count % 1000 == 0: print(f"Đã sinh {count}/{n_samples}...")

    print(f"Sinh dữ liệu xong: {time.time()-start:.1f}s")
    return np.array(X), np.array(y)

# --- CẢI TIẾN: MODEL MẠNH HƠN ---
def train_model(X, y):
    print("\n--- Training (Deep MLP) ---")
    
    # Scaler vẫn cần thiết ngay cả khi đã log
    s_X = StandardScaler()
    s_y = StandardScaler()
    X_s = s_X.fit_transform(X)
    y_s = s_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.1, random_state=42)
    
    # Tăng kích thước mạng: (256, 128, 64)
    # Vì input là 2700 chiều, layer đầu tiên nên lớn một chút
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,      # Batch size lớn giúp ổn định gradient
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        verbose=True         # Bật lên để xem Loss giảm thế nào
    )
    
    model.fit(X_train, y_train)
    
    # Đánh giá
    preds_s = model.predict(X_test)
    preds = s_y.inverse_transform(preds_s)
    real = s_y.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(real, preds, multioutput='raw_values'))
    print("\n>>> KẾT QUẢ CUỐI CÙNG (RMSE):")
    print(f"Vị trí X: {rmse[0]*100:.2f} cm")
    print(f"Vị trí Y: {rmse[1]*100:.2f} cm")
    print(f"Bán kính: {rmse[2]*100:.2f} cm")
    print(f"Chiều cao: {rmse[3]*100:.2f} cm")

if __name__ == "__main__":
    sim = VLPSimulator(ROOM_DIM, GRID_SIZE)
    X_data, y_data = generate_training_data(sim, N_SAMPLES)
    train_model(X_data, y_data)