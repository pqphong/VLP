import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. SYSTEM CONFIGURATION (CẤU HÌNH HỆ THỐNG)
# ==============================================================================
class VLPSystemConfig:
    def __init__(self):
        # Room dimensions (m)
        self.L, self.W, self.H = 5.0, 5.0, 3.0
        
        # Grid size 0.2m
        self.grid_step = 0.2  
        
        # Create coordinate grid
        self.x_range = np.arange(-self.L/2, self.L/2 + self.grid_step, self.grid_step)
        self.y_range = np.arange(-self.W/2, self.W/2 + self.grid_step, self.grid_step)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        
        # Input features: flattened arrays
        self.points_x = self.X.flatten()
        self.points_y = self.Y.flatten()
        
        # LED Configuration
        offset = 1.25
        self.led_coords = np.array([
            [offset, offset, self.H],
            [-offset, offset, self.H],
            [-offset, -offset, self.H],
            [offset, -offset, self.H]
        ])
        
        # Optical parameters
        self.P_total_per_led = 0.5
        self.m = 12.5                
        self.Adet = 1e-4
        self.Ts = 1.0
        self.n = 1.5
        self.FOV_rad = np.deg2rad(60.0)
        self.g_conc = (self.n**2) / (np.sin(self.FOV_rad)**2)
        
        # Diffuse component parameters
        self.rho = 0.8
        area_room = 2 * (self.L*self.W + self.L*self.H + self.W*self.H)
        self.H_diff = (self.rho * self.Adet) / (area_room * (1 - self.rho))

# ==============================================================================
# 2. POWER CALCULATION FUNCTION (HÀM TÍNH TOÁN)
# ==============================================================================
def calculate_power_profile(config, obj_params=None):
    # Initialize power array
    P_total_watts = np.zeros_like(config.points_x)
    
    # Check for object
    has_object = obj_params is not None
    if has_object:
        ox, oy = obj_params['x'], obj_params['y']
        oh, orad = obj_params['h'], obj_params['r']
    
    # Loop through LEDs
    for led in config.led_coords:
        lx, ly, lz = led
        
        # --- A. LoS Channel ---
        dx = config.points_x - lx
        dy = config.points_y - ly
        dz = 0.0 - lz 
        
        dist_sq = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist_sq)
        
        cos_phi = np.abs(dz) / dist
        cos_psi = cos_phi 
        
        # Lambertian formula
        H_los = ((config.m + 1) * config.Adet) / (2 * np.pi * dist_sq) * \
                (cos_phi ** config.m) * config.Ts * config.g_conc * cos_psi
        
        # FOV Clipping
        psi_angle = np.arccos(np.clip(cos_psi, -1.0, 1.0))
        H_los[psi_angle > config.FOV_rad] = 0.0
        
        # --- B. Shadowing ---
        if has_object:
            v_ray_x = config.points_x - lx
            v_ray_y = config.points_y - ly
            
            v_obj_x = ox - lx
            v_obj_y = oy - ly
            
            dot_prod = v_obj_x * v_ray_x + v_obj_y * v_ray_y
            ray_len_sq = v_ray_x**2 + v_ray_y**2
            t = dot_prod / (ray_len_sq + 1e-20)
            
            closest_x = lx + t * v_ray_x
            closest_y = ly + t * v_ray_y
            
            dist_perp = np.sqrt((closest_x - ox)**2 + (closest_y - oy)**2)
            
            cond_horiz = dist_perp < orad
            cond_order = (t > 0) & (t < 1)
            ray_height = lz * (1 - t)
            cond_vert = oh >= ray_height
            
            is_blocked = cond_horiz & cond_order & cond_vert
            H_los[is_blocked] = 0.0
            
        # Accumulate power
        P_total_watts += config.P_total_per_led * (H_los + config.H_diff)

    return P_total_watts

# ==============================================================================
# 3. DATA GENERATOR (HÀM SINH DỮ LIỆU)
# ==============================================================================
def generate_dataset(n_samples=100, mode='height'):
    cfg = VLPSystemConfig()
    
    # Calculate P0 (Empty room)
    print("Calculating Background Power Profile (P0)...")
    P0_watts = calculate_power_profile(cfg, obj_params=None)
    
    X_data = []
    y_data = []
    
    print(f"Generating {n_samples} samples (Mode: {mode.upper()})...")
    
    for i in range(n_samples):
        # Random object position
        ox = np.random.uniform(-2.0, 2.0)
        oy = np.random.uniform(-2.0, 2.0)
        
        if mode == 'height':
            oh = np.random.uniform(0.0, 2.0)
            orad = 0.05 
            label = oh
        elif mode == 'radius':
            oh = 1.0
            orad = np.random.uniform(0.0, 0.5)
            label = orad
        else:
            raise ValueError("Invalid Mode")
            
        # Calculate profile with object
        params = {'x': ox, 'y': oy, 'h': oh, 'r': orad}
        P_obs_watts = calculate_power_profile(cfg, params)
        
        # Calculate Differential Power
        diff_power = P0_watts - P_obs_watts
        
        # Append to list
        X_data.append(diff_power * 1e6) # Convert to uW
        y_data.append(label)
        
        if (i+1) % 50 == 0:
            print(f"  - Finished {i+1}/{n_samples} samples.")
            
    return np.array(X_data), np.array(y_data), cfg

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("--- START SIMULATION ---") # Đã sửa thành tiếng Anh để tránh lỗi Unicode
    
    # Generate 50 samples
    X, y, config = generate_dataset(n_samples=50, mode='height')
    
    print("\n--- DATA CHECK ---")
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    
    # Plot first sample
    idx = 0
    diff_map = X[idx].reshape(config.X.shape)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_map, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='jet')
    plt.colorbar(label='Differential Power (uW)')
    plt.title(f'Sample Differential Profile (Height={y[idx]:.2f}m)')
    plt.show()