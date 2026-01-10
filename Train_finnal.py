import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

# ==============================================================================
# 1. SYSTEM CONFIGURATION & PARAMETERS
# ==============================================================================
ROOM_DIM = [5, 5, 3]    # Length, Width, Height (m)
GRID_SIZE = 0.2         # Spatial Resolution (m)
# [IMPROVEMENT 1] Increased Sample Size for better Generalization without Leakage
N_SAMPLES = 50000       
MIN_VISIBLE_SENSORS = 1 # Minimum required LoS links

# Transmitter Parameters (LEDs)
LED_POWER = 10.0        # Radiated Power (Watts)
SEMI_ANGLE = 60         # Semi-angle at half power (Degrees)
LAMBERTIAN_ORDER = -np.log(2) / np.log(np.cos(np.radians(SEMI_ANGLE)))

# Receiver Parameters (Photodiodes)
PD_AREA = 1e-4          # Effective Active Area (m^2)
FOV = 60                # Field of View (Degrees)
REFRACTIVE_INDEX = 1.5
FILTER_GAIN = 1.0
CONC_GAIN = (REFRACTIVE_INDEX**2) / (np.sin(np.radians(FOV))**2)

# LED Constellation (Ceiling mounted at Z=3m)
LED_POSITIONS = np.array([
    [1.25, 1.25, 3.0], 
    [3.75, 1.25, 3.0],
    [3.75, 3.75, 3.0], 
    [1.25, 3.75, 3.0]
])

# ==============================================================================
# 2. PHYSICS-BASED SIMULATION ENGINE (VLP SIMULATOR)
# ==============================================================================
class VLPSimulator:
    def __init__(self, room_dim, grid_size):
        self.L, self.W, self.H = room_dim
        self.grid_size = grid_size
        
        # Initialize Sensor Grid
        x = np.arange(0, self.L + grid_size/100, grid_size)
        y = np.arange(0, self.W + grid_size/100, grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.rx_coords = np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel(), np.zeros(self.X_grid.size)))
        self.n_sensors = self.rx_coords.shape[0]
        print(f"[System Init] Sensor Grid Initialized: {self.n_sensors} nodes (Resolution: {grid_size}m)")

    def calculate_los_channel(self, led_pos, rx_pos_list, obj_params=None):
        """Calculates the DC Channel Gain (H0) considering geometric blockage."""
        vec_d = rx_pos_list - led_pos
        dist = np.linalg.norm(vec_d, axis=1)
        dist_sq = dist ** 2
        
        # Cosine of Irradiance (Phi) and Incidence (Psi) angles
        cos_phi = -vec_d[:, 2] / dist
        cos_psi = -vec_d[:, 2] / dist 
        
        H = np.zeros(len(dist))
        
        # FOV and Directionality Constraints
        fov_rad = np.radians(FOV)
        valid_indices = (cos_psi >= np.cos(fov_rad)) & (cos_phi > 0)
        
        const_factor = ((LAMBERTIAN_ORDER + 1) * PD_AREA) / (2 * np.pi)
        H[valid_indices] = (const_factor / dist_sq[valid_indices]) * \
                           (cos_phi[valid_indices] ** LAMBERTIAN_ORDER) * \
                           cos_psi[valid_indices] * FILTER_GAIN * CONC_GAIN

        # Shadowing / Blockage Logic (Cylindrical Model)
        if obj_params is not None:
            ox, oy, r, h = obj_params
            p1 = led_pos[:2]; p2 = rx_pos_list[:, :2]
            d_vec = p2 - p1; f_vec = p1 - np.array([ox, oy])
            
            # Intersection Math (Line vs Circle in 2D)
            a = np.sum(d_vec**2, axis=1)
            b = np.sum(d_vec * f_vec, axis=1)
            c = np.sum(f_vec**2) - r**2
            delta = b**2 - a*c
            pot_idx = np.where(delta >= 0)
            
            if len(pot_idx[0]) > 0:
                sqrt_delta = np.sqrt(delta[pot_idx])
                t1 = (-b[pot_idx] - sqrt_delta) / a[pot_idx]
                t2 = (-b[pot_idx] + sqrt_delta) / a[pot_idx]
                
                # Check segment intersection
                t_start = np.maximum(0, t1)
                t_end = np.minimum(1, t2)
                valid_intersect = t_start < t_end
                
                real_idx = pot_idx[0][valid_intersect]
                t_end_real = t_end[valid_intersect]
                
                # Height Check (Z-axis)
                z_at_exit = self.H * (1 - t_end_real)
                is_blocked = z_at_exit < h
                
                H[real_idx[is_blocked]] = 0.0
        return H

# ==============================================================================
# 3. DATA GENERATION PROTOCOL
# ==============================================================================
def generate_training_data(sim, n_samples):
    X = []
    y = []
    print(f"--- Initiating Data Generation Protocol ({n_samples} samples) ---")
    start_time = time.time()
    count = 0
    cos_threshold = np.cos(np.radians(SEMI_ANGLE))
    
    while count < n_samples:
        # Uniform Random Distribution for Object Parameters
        ox = np.random.uniform(0.5, sim.L - 0.5)
        oy = np.random.uniform(0.5, sim.W - 0.5)
        r = np.random.uniform(0.15, 0.40) 
        h = np.random.uniform(1.2, 1.9)   
        obj = [ox, oy, r, h]
        
        # Visibility Check (Heuristic)
        visible_led_count = 0
        obj_top = np.array([ox, oy, h])
        for led in LED_POSITIONS:
            vec = led - obj_top
            if abs(vec[2]) / np.linalg.norm(vec) > cos_threshold: visible_led_count += 1
        
        if visible_led_count < MIN_VISIBLE_SENSORS: continue 
            
        # RSS Simulation with AWGN Noise
        rss_features = []
        for led in LED_POSITIONS:
            h_channel = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_channel * LED_POWER
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            p_rx_noisy = np.maximum(p_rx + noise, 1e-12)
            
            # Logarithmic Transformation (dB) - Crucial for Neural Network
            p_rx_log = 10 * np.log10(p_rx_noisy) 
            rss_features.append(p_rx_log)
            
        X.append(np.concatenate(rss_features))
        y.append(obj)
        count += 1
        if count % 5000 == 0: print(f"   > Generated {count}/{n_samples} samples...")
            
    print(f"[Data Gen] Completed in {time.time() - start_time:.2f}s")
    return np.array(X), np.array(y)

# ==============================================================================
# 4. METHOD: DIFFERENTIAL POWER PROFILE (DPP)
# ==============================================================================
def calculate_p0_reference(sim):
    """
    Computes the Reference Power Profile (P0) for an Empty Room.
    Used as a baseline for the Differential Power Profile method.
    """
    print("--- [DPP Method] Computing Reference Empty Room Profile (P0)... ---")
    p0_total = np.zeros(sim.n_sensors)
    for led in LED_POSITIONS:
        h = sim.calculate_los_channel(led, sim.rx_coords, obj_params=None)
        p0_total += (h * LED_POWER)
    return p0_total

def predict_dpp_method(X_log_data, p0_ref, sim):
    """
    Implements the non-learning algorithm based on Differential Power Profile (DPP).
    """
    n_samples = X_log_data.shape[0]
    preds = []
    
    for i in range(n_samples):
        # Reconstruct Linear Power (Watts)
        sample_log = X_log_data[i].reshape(4, sim.n_sensors)
        sample_watts = 10**(sample_log / 10.0)
        
        # Aggregate and Compute Delta
        p_observed_total = np.sum(sample_watts, axis=0)
        diff_profile = p0_ref - p_observed_total
        
        # Peak Finding
        max_idx = np.argmax(diff_profile)
        pred_pos = sim.rx_coords[max_idx]
        
        preds.append([pred_pos[0], pred_pos[1]]) # Extract X, Y only
        
    return np.array(preds)

# ==============================================================================
# 5. COMPREHENSIVE VISUALIZATION SUITE (ACADEMIC STYLE)
# ==============================================================================

def plot_learning_curve(model, train_rmse, val_rmse, test_rmse):
    """Visualizes Training Convergence and Comparative RMSE."""
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_curve_, label='Training Loss', color='navy', linewidth=2)
    plt.title('Training Loss Convergence (Optimized)', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs'); plt.ylabel('Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    
    plt.subplot(1, 2, 2)
    sets = ['Train (80%)', 'Valid (10%)', 'Test (10%)']
    values = [train_rmse, val_rmse, test_rmse]
    bars = plt.bar(sets, values, color=['forestgreen', 'darkorange', 'firebrick'], alpha=0.9)
    plt.title('Performance Evaluation: RMSE Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE (cm)')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.2, 
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold')

def plot_matlab_style_regression(y_train_true, y_train_pred, y_val_true, y_val_pred, y_test_true, y_test_pred):
    """Regression Analysis (R-Value)."""
    print("\n--- Generating Regression Analysis Plots... ---")
    y_all_true = np.concatenate([y_train_true, y_val_true, y_test_true])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    datasets = [(y_train_true, y_train_pred, 'Training Set'), (y_val_true, y_val_pred, 'Validation Set'),
                (y_test_true, y_test_pred, 'Test Set'), (y_all_true, y_all_pred, 'Overall Dataset')]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    for i, (true, pred, title) in enumerate(datasets):
        ax = axs[i]
        t = true.flatten(); o = pred.flatten()
        R = np.corrcoef(t, o)[0, 1] if len(t) > 1 else 0
        slope, intercept = np.polyfit(t, o, 1) if len(t) > 1 else (1, 0)
        
        ax.scatter(t, o, facecolors='none', edgecolors='k', s=15, label='Data Points')
        x_vals = np.array([min(t), max(t)])
        ax.plot(x_vals, slope * x_vals + intercept, color=['blue', 'green', 'red', 'black'][i], linewidth=2, label=f'Fit: R={R:.4f}')
        ax.plot([min(t), max(t)], [min(t), max(t)], 'k-o', alpha=0.5, label='Y = T (Ideal)')
        ax.set_title(f'{title}', fontweight='bold')
        ax.set_xlabel('Target Value'); ax.set_ylabel('Output Value'); ax.legend(loc='upper left'); ax.grid(True, linestyle=':')
    plt.tight_layout()

def plot_parameter_estimation(y_true, y_pred):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 2], y_pred[:, 2], alpha=0.6, color='purple', s=15)
    plt.plot([0.15, 0.4], [0.15, 0.4], 'k--', lw=2, label='Ground Truth')
    rmse_r = np.sqrt(mean_squared_error(y_true[:,2], y_pred[:,2])) * 100
    plt.title(f'Radius Accuracy (RMSE: {rmse_r:.2f} cm)', fontweight='bold')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 3], y_pred[:, 3], alpha=0.6, color='brown', s=15)
    plt.plot([1.2, 1.9], [1.2, 1.9], 'k--', lw=2, label='Ground Truth')
    rmse_h = np.sqrt(mean_squared_error(y_true[:,3], y_pred[:,3])) * 100
    plt.title(f'Height Accuracy (RMSE: {rmse_h:.2f} cm)', fontweight='bold')
    plt.grid(True)

def plot_cdf_error(y_true, y_pred):
    errors = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    errors_sorted = np.sort(errors)
    p = 1. * np.arange(len(errors)) / (len(errors) - 1)
    p90 = np.percentile(errors, 90)
    plt.figure(figsize=(8, 6))
    plt.plot(errors_sorted * 100, p, linewidth=2.5, color='darkcyan')
    plt.axvline(x=p90*100, color='r', linestyle='--', label=f'90% Confidence < {p90*100:.1f} cm')
    plt.title('Cumulative Distribution Function (CDF) of Error', fontweight='bold')
    plt.xlabel('Positioning Error (cm)'); plt.ylabel('Probability'); plt.grid(True); plt.legend()

def visualize_trajectory(sim, model, scaler_X, scaler_y):
    print("\n--- Visualizing Trajectory Tracking (Proposed DNN)... ---")
    theta = np.linspace(0, 2*np.pi, 40)
    path_x = 2.5 + 1.5 * np.cos(theta); path_y = 2.5 + 1.5 * np.sin(theta)
    pred_path = []
    
    for i in range(len(path_x)):
        obj = [path_x[i], path_y[i], 0.3, 1.6]
        rss_features = []
        for led in LED_POSITIONS:
            h_val = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_val * LED_POWER
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            rss_features.append(10 * np.log10(np.maximum(p_rx + noise, 1e-12)))
        X_in = scaler_X.transform(np.concatenate(rss_features).reshape(1, -1))
        pred_path.append(scaler_y.inverse_transform(model.predict(X_in))[0, :2])
        
    pred_path = np.array(pred_path)
    plt.figure(figsize=(6, 6))
    plt.plot(path_x, path_y, 'g-o', linewidth=2, label='Ground Truth')
    plt.plot(pred_path[:, 0], pred_path[:, 1], 'b-o', markersize=5, label='Predicted (DNN)')
    plt.xlim(0, 5); plt.ylim(0, 5)
    plt.title('Trajectory Tracking Performance', fontweight='bold')
    plt.legend(); plt.grid(True)

def stress_test(sim, model, scaler_X, scaler_y):
    print("\n--- Executing Robustness Analysis (Stress Test)... ---")
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
    
    plt.figure(figsize=(7, 4))
    plt.plot([str(x) for x in noises], rmses, 'r-o', linewidth=2)
    plt.title('Robustness Analysis: RMSE vs Noise Level', fontweight='bold')
    plt.xlabel('Noise Standard Deviation (W)'); plt.ylabel('Mean RMSE (cm)'); plt.grid(True)

def plot_comparison_result(dl_rmse, dpp_rmse):
    plt.figure(figsize=(9, 6))
    names = ['Proposed DNN Model', 'Differential Power Profile (DPP)']
    values = [dl_rmse, dpp_rmse]
    colors = ['royalblue', 'indianred'] 
    bars = plt.bar(names, values, color=colors, width=0.6, alpha=0.9)
    plt.title('Comparative Analysis: Positioning Accuracy (RMSE)', fontweight='bold')
    plt.ylabel('RMSE (cm)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{bar.get_height():.2f} cm', ha='center', fontweight='bold', fontsize=11)

def visualize_trajectory_comparison(sim, dl_model, p0_ref, scaler_X, scaler_y):
    print("--- Generating Comparative Trajectory Analysis... ---")
    theta = np.linspace(0, 2*np.pi, 40)
    path_x = 2.5 + 1.5 * np.cos(theta); path_y = 2.5 + 1.5 * np.sin(theta)
    
    pred_dl = []
    pred_dpp = []
    
    for i in range(len(path_x)):
        obj = [path_x[i], path_y[i], 0.3, 1.6]
        rss_features = []
        for led in LED_POSITIONS:
            h_val = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_val * LED_POWER
            noise = np.random.normal(0, 1e-8, size=len(p_rx))
            rss_features.append(10 * np.log10(np.maximum(p_rx + noise, 1e-12)))
        
        X_log_sample = np.concatenate(rss_features)
        
        # Predict DL
        X_in = scaler_X.transform(X_log_sample.reshape(1, -1))
        pred_dl.append(scaler_y.inverse_transform(dl_model.predict(X_in))[0, :2])
        
        # Predict DPP
        pred_p = predict_dpp_method(np.array([X_log_sample]), p0_ref, sim)
        pred_dpp.append(pred_p[0])
        
    pred_dl = np.array(pred_dl); pred_dpp = np.array(pred_dpp)
    
    plt.figure(figsize=(8, 8))
    plt.plot(path_x, path_y, 'k-o', linewidth=2.5, label='Ground Truth')
    plt.plot(pred_dl[:, 0], pred_dl[:, 1], 'b-o', markersize=6, label='Proposed DNN')
    plt.plot(pred_dpp[:, 0], pred_dpp[:, 1], 'r-^', markersize=6, label='Differential Power Profile (DPP)')
    plt.xlim(0, 5); plt.ylim(0, 5)
    plt.title('Comparative Trajectory Tracking', fontweight='bold')
    plt.legend(); plt.grid(True)

# ==============================================================================
# 7. MAIN EXECUTION PROTOCOL (OPTIMIZED)
# ==============================================================================
def train_and_evaluate(X, y, sim):
    print("\n=======================================================")
    print("   COMMENCING EXPERIMENTAL PROTOCOL (OPTIMIZED)   ")
    print("=======================================================")
    
    # 1. Correct Data Split (No Leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_raw, y_train, test_size=0.1111, random_state=42)
    
    # 2. Scaler fitted ONLY on Train (Crucial for Integrity)
    s_X = StandardScaler().fit(X_train_raw)
    s_y = StandardScaler().fit(y_train)
    
    X_train_s = s_X.transform(X_train_raw)
    X_val_s = s_X.transform(X_val_raw)
    X_test_s = s_X.transform(X_test_raw)
    y_train_s = s_y.transform(y_train)
    y_val_s = s_y.transform(y_val)
    
    print(f"[Dataset Stats] Train: {len(X_train_raw)} | Valid: {len(X_val_raw)} | Test: {len(X_test_raw)}")
    
    # --- PHASE A: DEEP LEARNING MODEL (HYPER-TUNED) ---
    print("\n[Phase A] Training Optimized Deep Neural Network...")
    # [IMPROVEMENT 2]: Optimized Hyperparameters
    dl_model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64), # Deeper and Wider
        activation='relu', 
        solver='adam',
        alpha=0.1,                # Increased Regularization (Prevents Overfitting)
        batch_size=1024,              # Smaller Batch Size (Better generalization)
        learning_rate='adaptive',   # Adaptive Learning Rate (Better convergence)
        learning_rate_init=0.001,
        max_iter=500, 
        early_stopping=True, 
        n_iter_no_change=20,        # More patience
        verbose=True
    )
    dl_model.fit(X_train_s, y_train_s)
    
    # --- PHASE B: DIFFERENTIAL POWER PROFILE (DPP) ---
    print("\n[Phase B] Executing Differential Power Profile Method (DPP)...")
    p0_ref = calculate_p0_reference(sim)
    dpp_pred = predict_dpp_method(X_test_raw, p0_ref, sim) # Uses Raw Log Data
    
    # --- PHASE C: EVALUATION ---
    dl_pred_s = dl_model.predict(X_test_s)
    dl_pred = s_y.inverse_transform(dl_pred_s)
    
    real_xy = y_test[:, :2]
    
    def calc_rmse(real, pred): return np.sqrt(np.mean(np.sum((real-pred)**2, axis=1))) * 100
    
    dl_rmse = calc_rmse(real_xy, dl_pred[:, :2])
    dpp_rmse = calc_rmse(real_xy, dpp_pred)
    
    print("\n>>> FINAL PERFORMANCE METRICS (RMSE):")
    print(f" 1. Proposed DNN Model       : {dl_rmse:.2f} cm")
    print(f" 2. Diff. Power Profile      : {dpp_rmse:.2f} cm")
    
    # --- VISUAL ANALYTICS ---
    print("\n--- Generating Visual Analytics... ---")
    y_tr_pred = s_y.inverse_transform(dl_model.predict(X_train_s))
    y_va_pred = s_y.inverse_transform(dl_model.predict(X_val_s))
    
    rmse_tr = calc_rmse(y_train[:, :2], y_tr_pred[:, :2])
    rmse_va = calc_rmse(y_val[:, :2], y_va_pred[:, :2])
    
    plot_learning_curve(dl_model, rmse_tr, rmse_va, dl_rmse)
    plot_matlab_style_regression(y_train, y_tr_pred, y_val, y_va_pred, y_test, dl_pred)
    plot_parameter_estimation(y_test, dl_pred)
    plot_cdf_error(y_test, dl_pred)
    visualize_trajectory(sim, dl_model, s_X, s_y)
    stress_test(sim, dl_model, s_X, s_y)
    
    plot_comparison_result(dl_rmse, dpp_rmse)
    visualize_trajectory_comparison(sim, dl_model, p0_ref, s_X, s_y)

if __name__ == "__main__":
    sim = VLPSimulator(ROOM_DIM, GRID_SIZE)
    # Generate Data (Increased Size)
    X_data, y_data = generate_training_data(sim, N_SAMPLES)
    # Execute Main Protocol
    train_and_evaluate(X_data, y_data, sim)
    plt.show()