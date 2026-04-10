import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
# 4. COMPREHENSIVE VISUALIZATION SUITE (ACADEMIC STYLE)
# ==============================================================================

def draw_room_outline_3d(ax, sim):
    """Draws a simple 3D room wireframe."""
    room_x = [0, sim.L, sim.L, 0, 0]
    room_y = [0, 0, sim.W, sim.W, 0]
    ax.plot(room_x, room_y, [0] * len(room_x), 'k--', alpha=0.45, linewidth=1)
    ax.plot(room_x, room_y, [sim.H] * len(room_x), 'k--', alpha=0.20, linewidth=1)

    corners = [(0, 0), (sim.L, 0), (sim.L, sim.W), (0, sim.W)]
    for corner_x, corner_y in corners:
        ax.plot([corner_x, corner_x], [corner_y, corner_y], [0, sim.H], 'k--', alpha=0.15, linewidth=1)

def draw_pd_layout_2d(ax, sim, color='white', alpha=0.30, size=8, label=None):
    """Draws the PD receiver grid on the floor plane in 2D."""
    ax.scatter(
        sim.rx_coords[:, 0],
        sim.rx_coords[:, 1],
        marker='o',
        s=size,
        color=color,
        alpha=alpha,
        linewidths=0,
        label=label
    )

def draw_pd_layout_3d(ax, sim, color='gray', alpha=0.18, size=9, label=None):
    """Draws the PD receiver grid on the floor plane in 3D."""
    ax.scatter(
        sim.rx_coords[:, 0],
        sim.rx_coords[:, 1],
        sim.rx_coords[:, 2],
        marker='o',
        s=size,
        color=color,
        alpha=alpha,
        depthshade=False,
        label=label
    )

def draw_cylinder_3d(ax, center_x, center_y, radius, height, color, alpha=0.35, resolution=40, label=None):
    """Draws a vertical cylinder with bottom and top caps."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = center_x + radius * np.cos(theta_grid)
    y_grid = center_y + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0, shade=True)

    radial = np.linspace(0, radius, resolution // 2)
    radial_grid, theta_cap = np.meshgrid(radial, theta)
    x_cap = center_x + radial_grid * np.cos(theta_cap)
    y_cap = center_y + radial_grid * np.sin(theta_cap)
    z_top = np.full_like(x_cap, height)
    z_bottom = np.zeros_like(x_cap)
    ax.plot_surface(x_cap, y_cap, z_top, color=color, alpha=alpha, linewidth=0, shade=True)
    ax.plot_surface(x_cap, y_cap, z_bottom, color=color, alpha=min(alpha * 0.6, 0.25), linewidth=0, shade=False)

    if label is not None:
        ax.plot([], [], [], color=color, linewidth=8, alpha=alpha, label=label)

def visualize_sensor_layout(sim):
    """Visualizes the spatial layout of the PD receiver grid and LEDs."""
    print("\n--- Visualizing PD and LED Layout... ---")

    fig = plt.figure(figsize=(14, 6))
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

    draw_pd_layout_2d(ax_2d, sim, color='dimgray', alpha=0.55, size=14, label='PD receiver grid')
    ax_2d.scatter(
        LED_POSITIONS[:, 0],
        LED_POSITIONS[:, 1],
        marker='^',
        s=130,
        color='crimson',
        edgecolors='black',
        label='LED positions'
    )
    ax_2d.set_title('2D Sensor Layout', fontweight='bold')
    ax_2d.set_xlabel('X position (m)')
    ax_2d.set_ylabel('Y position (m)')
    ax_2d.set_xlim(0, sim.L)
    ax_2d.set_ylim(0, sim.W)
    ax_2d.grid(True, linestyle=':', alpha=0.5)
    ax_2d.legend(loc='upper right')

    draw_room_outline_3d(ax_3d, sim)
    draw_pd_layout_3d(ax_3d, sim, color='dimgray', alpha=0.35, size=10, label='PD receiver grid')
    ax_3d.scatter(
        LED_POSITIONS[:, 0],
        LED_POSITIONS[:, 1],
        LED_POSITIONS[:, 2],
        marker='^',
        s=140,
        color='crimson',
        edgecolors='black',
        label='LED positions'
    )
    ax_3d.set_title('3D Sensor Layout', fontweight='bold')
    ax_3d.set_xlabel('X position (m)')
    ax_3d.set_ylabel('Y position (m)')
    ax_3d.set_zlabel('Z position (m)')
    ax_3d.set_xlim(0, sim.L)
    ax_3d.set_ylim(0, sim.W)
    ax_3d.set_zlim(0, sim.H)
    ax_3d.view_init(elev=24, azim=-58)
    ax_3d.legend(loc='upper left')

    plt.suptitle(f'PD Grid and LED Layout | Total PDs: {sim.n_sensors}', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def visualize_sample_geometry_3d(sim, y, sample_index=0):
    """Visualizes one generated object as a true 3D cylinder inside the room."""
    print("\n--- Visualizing Sample Geometry in 3D... ---")

    if len(y) == 0:
        print("[Visualization] Skipped because the generated dataset is empty.")
        return

    sample_index = int(np.clip(sample_index, 0, len(y) - 1))
    obj_x, obj_y, obj_r, obj_h = y[sample_index]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    draw_room_outline_3d(ax, sim)
    draw_pd_layout_3d(ax, sim, color='dimgray', alpha=0.25, size=8, label='PD receiver grid')
    draw_cylinder_3d(
        ax,
        obj_x,
        obj_y,
        obj_r,
        obj_h,
        color='deepskyblue',
        alpha=0.40,
        label='Object cylinder'
    )

    ax.scatter(
        LED_POSITIONS[:, 0],
        LED_POSITIONS[:, 1],
        LED_POSITIONS[:, 2],
        marker='^',
        s=140,
        color='crimson',
        edgecolors='black',
        label='LED positions'
    )
    ax.scatter(
        [obj_x],
        [obj_y],
        [obj_h],
        marker='o',
        s=50,
        color='navy',
        label='Object top center'
    )

    ax.set_title('3D Room View of Example Object', fontweight='bold')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_xlim(0, sim.L)
    ax.set_ylim(0, sim.W)
    ax.set_zlim(0, sim.H)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc='upper left')

    plt.suptitle(
        f'Example Sample #{sample_index} | x={obj_x:.2f} m, y={obj_y:.2f} m, '
        f'r={obj_r:.2f} m, h={obj_h:.2f} m',
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def visualize_generated_data(sim, X, y, sample_index=0):
    """Visualizes the generated dataset before model training."""
    print("\n--- Visualizing Generated Dataset... ---")

    if len(X) == 0 or len(y) == 0:
        print("[Visualization] Skipped because the generated dataset is empty.")
        return

    n_leds = len(LED_POSITIONS)
    sensor_count = sim.n_sensors
    sample_index = int(np.clip(sample_index, 0, len(X) - 1))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    center_density = axs[0, 0].hexbin(
        y[:, 0],
        y[:, 1],
        gridsize=22,
        extent=(0, sim.L, 0, sim.W),
        cmap='viridis',
        mincnt=1
    )
    fig.colorbar(center_density, ax=axs[0, 0], label='Samples per bin')
    draw_pd_layout_2d(axs[0, 0], sim, color='white', alpha=0.28, size=10, label='PD receiver grid')
    axs[0, 0].scatter(
        LED_POSITIONS[:, 0],
        LED_POSITIONS[:, 1],
        marker='^',
        s=120,
        color='crimson',
        edgecolors='black',
        label='LED positions'
    )
    axs[0, 0].set_title('Object Center Distribution with PD Grid', fontweight='bold')
    axs[0, 0].set_xlabel('X position (m)')
    axs[0, 0].set_ylabel('Y position (m)')
    axs[0, 0].set_xlim(0, sim.L)
    axs[0, 0].set_ylim(0, sim.W)
    axs[0, 0].grid(True, linestyle=':', alpha=0.5)
    axs[0, 0].legend()

    axs[0, 1].hist(y[:, 2] * 100, bins=30, color='slateblue', edgecolor='black', alpha=0.85)
    axs[0, 1].set_title('Radius Distribution', fontweight='bold')
    axs[0, 1].set_xlabel('Radius (cm)')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].grid(True, linestyle=':', alpha=0.5)

    axs[1, 0].hist(y[:, 3], bins=30, color='peru', edgecolor='black', alpha=0.85)
    axs[1, 0].set_title('Height Distribution', fontweight='bold')
    axs[1, 0].set_xlabel('Height (m)')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True, linestyle=':', alpha=0.5)

    led_mean_rss = [
        X[:, led_idx * sensor_count:(led_idx + 1) * sensor_count].mean(axis=1)
        for led_idx in range(n_leds)
    ]
    axs[1, 1].boxplot(led_mean_rss, labels=[f'LED {i+1}' for i in range(n_leds)], patch_artist=True)
    axs[1, 1].set_title('Mean RSS per LED Across Samples', fontweight='bold')
    axs[1, 1].set_ylabel('Mean RSS (dB)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()

    sample_maps = X[sample_index].reshape(n_leds, sim.X_grid.shape[0], sim.X_grid.shape[1])
    obj_x, obj_y, obj_r, obj_h = y[sample_index]

    fig = plt.figure(figsize=(14, 10))
    axs = [fig.add_subplot(2, 2, idx + 1, projection='3d') for idx in range(n_leds)]
    for led_idx, ax in enumerate(axs):
        rss_map = sample_maps[led_idx]
        surface = ax.plot_surface(
            sim.X_grid,
            sim.Y_grid,
            rss_map,
            cmap='inferno',
            linewidth=0,
            antialiased=True,
            alpha=0.95
        )

        obj_grid_idx = np.unravel_index(
            np.argmin((sim.X_grid - obj_x) ** 2 + (sim.Y_grid - obj_y) ** 2),
            sim.X_grid.shape
        )
        led_grid_idx = np.unravel_index(
            np.argmin(
                (sim.X_grid - LED_POSITIONS[led_idx, 0]) ** 2 +
                (sim.Y_grid - LED_POSITIONS[led_idx, 1]) ** 2
            ),
            sim.X_grid.shape
        )

        obj_rss = rss_map[obj_grid_idx]
        led_rss = rss_map[led_grid_idx]

        ax.scatter(
            obj_x,
            obj_y,
            obj_rss + 1.5,
            marker='x',
            s=90,
            color='cyan',
            linewidths=2.5,
            label='Object center'
        )
        ax.scatter(
            LED_POSITIONS[led_idx, 0],
            LED_POSITIONS[led_idx, 1],
            led_rss + 1.5,
            marker='^',
            s=120,
            color='white',
            edgecolors='black',
            label='LED XY projection'
        )

        ax.set_title(f'LED {led_idx + 1} RSS Surface', fontweight='bold')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('RSS (dB)')
        ax.set_xlim(0, sim.L)
        ax.set_ylim(0, sim.W)
        ax.set_zlim(rss_map.min() - 2, rss_map.max() + 3)
        ax.view_init(elev=28, azim=-135)
        if led_idx == 0:
            ax.legend(loc='upper right')
        fig.colorbar(surface, ax=ax, fraction=0.046, pad=0.06, label='RSS (dB)')

    plt.suptitle(
        f'Example Generated Sample #{sample_index} | x={obj_x:.2f} m, y={obj_y:.2f} m, '
        f'r={obj_r:.2f} m, h={obj_h:.2f} m',
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

def visualize_generated_data_3d(sim, X, y, max_points=3000, sample_index=0):
    """Visualizes the generated dataset in 3D space."""
    print("\n--- Visualizing Generated Dataset in 3D... ---")

    if len(X) == 0 or len(y) == 0:
        print("[Visualization] Skipped because the generated dataset is empty.")
        return

    rng = np.random.default_rng(42)
    if len(y) > max_points:
        subset_idx = rng.choice(len(y), size=max_points, replace=False)
    else:
        subset_idx = np.arange(len(y))

    X_subset = X[subset_idx]
    y_subset = y[subset_idx]
    mean_rss_subset = X_subset.mean(axis=1)

    sample_index = int(np.clip(sample_index, 0, len(y) - 1))
    obj_x, obj_y, obj_r, obj_h = y[sample_index]

    fig = plt.figure(figsize=(15, 6))
    ax_radius = fig.add_subplot(1, 2, 1, projection='3d')
    ax_rss = fig.add_subplot(1, 2, 2, projection='3d')

    radius_plot = ax_radius.scatter(
        y_subset[:, 0],
        y_subset[:, 1],
        y_subset[:, 3],
        c=y_subset[:, 2] * 100,
        cmap='plasma',
        s=10,
        alpha=0.70
    )
    ax_radius.scatter(
        LED_POSITIONS[:, 0],
        LED_POSITIONS[:, 1],
        LED_POSITIONS[:, 2],
        marker='^',
        s=130,
        color='crimson',
        edgecolors='black',
        label='LED positions'
    )
    draw_room_outline_3d(ax_radius, sim)
    draw_pd_layout_3d(ax_radius, sim, color='dimgray', alpha=0.15, size=8, label='PD receiver grid')
    draw_cylinder_3d(
        ax_radius,
        obj_x,
        obj_y,
        obj_r,
        obj_h,
        color='deepskyblue',
        alpha=0.40,
        label='Highlighted object'
    )
    ax_radius.set_title('3D Sample Distribution Colored by Radius', fontweight='bold')
    ax_radius.set_xlabel('X position (m)')
    ax_radius.set_ylabel('Y position (m)')
    ax_radius.set_zlabel('Object height (m)')
    ax_radius.set_xlim(0, sim.L)
    ax_radius.set_ylim(0, sim.W)
    ax_radius.set_zlim(0, sim.H)
    ax_radius.view_init(elev=24, azim=-58)
    ax_radius.legend(loc='upper left')
    fig.colorbar(radius_plot, ax=ax_radius, fraction=0.046, pad=0.08, label='Radius (cm)')

    rss_plot = ax_rss.scatter(
        y_subset[:, 0],
        y_subset[:, 1],
        y_subset[:, 3],
        c=mean_rss_subset,
        cmap='viridis',
        s=10,
        alpha=0.70
    )
    ax_rss.scatter(
        LED_POSITIONS[:, 0],
        LED_POSITIONS[:, 1],
        LED_POSITIONS[:, 2],
        marker='^',
        s=130,
        color='crimson',
        edgecolors='black'
    )
    draw_room_outline_3d(ax_rss, sim)
    draw_pd_layout_3d(ax_rss, sim, color='dimgray', alpha=0.15, size=8)
    draw_cylinder_3d(
        ax_rss,
        obj_x,
        obj_y,
        obj_r,
        obj_h,
        color='deepskyblue',
        alpha=0.40,
        label='Highlighted object'
    )
    ax_rss.set_title('3D Sample Distribution Colored by Mean RSS', fontweight='bold')
    ax_rss.set_xlabel('X position (m)')
    ax_rss.set_ylabel('Y position (m)')
    ax_rss.set_zlabel('Object height (m)')
    ax_rss.set_xlim(0, sim.L)
    ax_rss.set_ylim(0, sim.W)
    ax_rss.set_zlim(0, sim.H)
    ax_rss.view_init(elev=24, azim=-58)
    fig.colorbar(rss_plot, ax=ax_rss, fraction=0.046, pad=0.08, label='Mean RSS (dB)')

    plt.suptitle(
        f'3D Overview of Generated Samples ({len(subset_idx)} of {len(y)} points shown)',
        fontweight='bold'
    )
    plt.tight_layout()

def plot_learning_curve(model, train_rmse, val_rmse, test_rmse):
    """Visualizes training convergence and RMSE across data splits."""
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

# ==============================================================================
# 5. MAIN EXECUTION PROTOCOL (OPTIMIZED)
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
    
    # --- PHASE B: EVALUATION ---
    dl_pred_s = dl_model.predict(X_test_s)
    dl_pred = s_y.inverse_transform(dl_pred_s)
    
    real_xy = y_test[:, :2]
    
    def calc_rmse(real, pred): return np.sqrt(np.mean(np.sum((real-pred)**2, axis=1))) * 100
    
    dl_rmse = calc_rmse(real_xy, dl_pred[:, :2])
    
    print("\n>>> FINAL PERFORMANCE METRIC (RMSE):")
    print(f" Proposed DNN Model: {dl_rmse:.2f} cm")
    
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

if __name__ == "__main__":
    sim = VLPSimulator(ROOM_DIM, GRID_SIZE)
    # Generate Data (Increased Size)
    X_data, y_data = generate_training_data(sim, N_SAMPLES)
    visualize_sensor_layout(sim)
    visualize_generated_data(sim, X_data, y_data, sample_index=0)
    visualize_sample_geometry_3d(sim, y_data, sample_index=0)
    visualize_generated_data_3d(sim, X_data, y_data, max_points=3000, sample_index=0)
    # Execute Main Protocol
    train_and_evaluate(X_data, y_data, sim)
    plt.show()
