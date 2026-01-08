# Visible Light Positioning (VLP) simulation and neural network training for object shape estimation

# ---------------------------------------------
# Import necessary libraries
# ---------------------------------------------
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------
# Environment and simulation parameters
# ---------------------------------------------
room_length = 5.0  # room dimensions 5m x 5m
room_width = 5.0
LED_height = 3.0   # ceiling height 3m
# Positions of 4 LED light sources (at ceiling, positioned at quarter-points of room)
LED_positions = [(1.25, 1.25, LED_height),
                 (1.25, 3.75, LED_height),
                 (3.75, 1.25, LED_height),
                 (3.75, 3.75, LED_height)]
m = 1              # Lambertian order (for LED radiation pattern)
P_tx = 100.0       # LED transmit optical power (relative units)
A_pd = 1.0         # photodetector effective area (assume 1 for simplicity)

# Photodetector grid: 26x26 points (676 PD) covering room floor with 0.2m spacing
grid_spacing = 0.2
points_per_axis = int(room_length / grid_spacing) + 1  # = 26
xs = np.linspace(0, room_length, points_per_axis)
ys = np.linspace(0, room_width, points_per_axis)
xx, yy = np.meshgrid(xs, ys)
xx_flat = xx.flatten()
yy_flat = yy.flatten()
num_pd = xx_flat.size  # 676

# ---------------------------------------------
# Simulation function for received power profile
# ---------------------------------------------
def simulate_power_profile(x_c, y_c, R, H):
    """
    Simulate the received power profile on the PD grid for a cylindrical object 
    at position (x_c, y_c) with radius R and height H. Returns a 1D array of length 676.
    """
    total_power = np.zeros(num_pd)
    for (lx, ly, lz) in LED_positions:
        # Compute direct illumination (no blocking) from this LED to all PD points
        dx = xx_flat - lx
        dy = yy_flat - ly
        dist_horiz_sq = dx**2 + dy**2                  # horizontal distance squared
        dz = lz                                        # vertical distance (LED height)
        d = np.sqrt(dist_horiz_sq + dz**2)             # 3D distance from LED to PD
        cos_phi = dz / d                               # cosine of incidence angle
        cos_factor = cos_phi ** (m+1)                  # Lambertian (cos^m) * receiver gain (cos)
        base_intensity = P_tx * (m+1)/(2*np.pi) * A_pd * cos_factor / (d**2)
        # If no object (R=0 or H=0), no blocking:
        if R <= 0 or H <= 0:
            total_power += base_intensity
            continue
        # Check line-of-sight blockage by the cylinder for each PD
        cx = x_c - lx
        cy = y_c - ly
        # Projection parameter t_h of cylinder center onto the line from LED to PD (in horizontal plane)
        dot = cx * dx + cy * dy
        t_h = np.zeros_like(dx)
        mask_nonzero = dist_horiz_sq > 1e-12
        t_h[mask_nonzero] = dot[mask_nonzero] / dist_horiz_sq[mask_nonzero]
        # Determine which PDs have the cylinder directly between them and the LED
        mask_segment = (t_h >= 0) & (t_h <= 1)         # cylinder is horizontally between LED and PD
        # Special case: PD directly below LED (horizontal distance ~ 0)
        mask_vert = dist_horiz_sq < 1e-12
        if np.any(mask_vert):
            dist_center = np.sqrt(cx**2 + cy**2)       # horizontal distance from cylinder center to LED-PD line
            if dist_center <= R:
                # Cylinder covers the vertical line-of-sight
                base_intensity[mask_vert] = 0.0
            # Exclude these from further blocking checks
            mask_segment &= ~mask_vert
        # For PDs where cylinder is between LED and PD, check distance and height
        if np.any(mask_segment):
            # Shortest distance from cylinder center to the line (in horizontal plane)
            cross_val = dx * cy - dy * cx
            dist_line = np.abs(cross_val[mask_segment]) / (np.sqrt(dist_horiz_sq[mask_segment]) + 1e-12)
            # PDs where the line-of-sight passes within the cylinder's radius
            mask_close = dist_line <= R
            if np.any(mask_close):
                # Height of line-of-sight at the cylinder's horizontal position
                z_line = lz * (1 - t_h[mask_segment][mask_close])  # lz = LED height
                # If the cylinder's height covers this line, it blocks the light
                mask_block = z_line <= H
                # Zero out contributions for blocked PDs
                blocked_indices = np.where(mask_segment)[0][mask_close][mask_block]
                base_intensity[blocked_indices] = 0.0
        # Accumulate contribution of this LED (with blocking)
        total_power += base_intensity
    return total_power

# ---------------------------------------------
# Generate dataset for height estimation (Task 1)
# ---------------------------------------------
N_samples = 1000
fixed_radius = 0.05  # use a small fixed radius for height estimation task
X_height = np.zeros((N_samples, num_pd))
y_height = np.zeros(N_samples)
for i in range(N_samples):
    H = np.random.uniform(0.0, 2.0)       # random height between 0 and 2 m
    R = fixed_radius
    # Random object position, keeping entire cylinder inside room
    if R > 0:
        x_c = np.random.uniform(R, room_length - R)
        y_c = np.random.uniform(R, room_width - R)
    else:
        x_c = np.random.uniform(0.0, room_length)
        y_c = np.random.uniform(0.0, room_width)
    # Simulate the received power profile for this sample
    X_height[i] = simulate_power_profile(x_c, y_c, R, H)
    y_height[i] = H
# Save height dataset to .npz file
np.savez("height_dataset.npz", X=X_height, y=y_height)

# Generate dataset for radius estimation (Task 2)
fixed_height = 1.0   # use a fixed height for radius estimation task
X_radius = np.zeros((N_samples, num_pd))
y_radius = np.zeros(N_samples)
for i in range(N_samples):
    R = np.random.uniform(0.0, 0.5)       # random radius between 0 and 0.5 m
    H = fixed_height
    if R > 0:
        x_c = np.random.uniform(R, room_length - R)
        y_c = np.random.uniform(R, room_width - R)
    else:
        x_c = np.random.uniform(0.0, room_length)
        y_c = np.random.uniform(0.0, room_width)
    X_radius[i] = simulate_power_profile(x_c, y_c, R, H)
    y_radius[i] = R
np.savez("radius_dataset.npz", X=X_radius, y=y_radius)

# ---------------------------------------------
# Split into train/validation/test sets (80/10/10)
# ---------------------------------------------
def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1):
    N = X.shape[0]
    indices = np.random.permutation(N)
    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)
    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

Xh_train, yh_train, Xh_val, yh_val, Xh_test, yh_test = split_dataset(X_height, y_height)
Xr_train, yr_train, Xr_val, yr_val, Xr_test, yr_test = split_dataset(X_radius, y_radius)

# Normalize targets to [0,1] for training (scale height by 2, radius by 0.5)
yh_train_scaled = yh_train / 2.0
yh_val_scaled   = yh_val   / 2.0
yh_test_scaled  = yh_test  / 2.0
yr_train_scaled = yr_train / 0.5
yr_val_scaled   = yr_val   / 0.5
yr_test_scaled  = yr_test  / 0.5

# Convert to PyTorch tensors
Xh_train_t = torch.from_numpy(Xh_train).float()
Xh_val_t   = torch.from_numpy(Xh_val).float()
Xh_test_t  = torch.from_numpy(Xh_test).float()
yh_train_t = torch.from_numpy(yh_train_scaled).float().unsqueeze(1)
yh_val_t   = torch.from_numpy(yh_val_scaled).float().unsqueeze(1)
yh_test_t  = torch.from_numpy(yh_test_scaled).float().unsqueeze(1)

Xr_train_t = torch.from_numpy(Xr_train).float()
Xr_val_t   = torch.from_numpy(Xr_val).float()
Xr_test_t  = torch.from_numpy(Xr_test).float()
yr_train_t = torch.from_numpy(yr_train_scaled).float().unsqueeze(1)
yr_val_t   = torch.from_numpy(yr_val_scaled).float().unsqueeze(1)
yr_test_t  = torch.from_numpy(yr_test_scaled).float().unsqueeze(1)

# Create DataLoader for batching
batch_size = 32
train_loader_h = DataLoader(TensorDataset(Xh_train_t, yh_train_t), batch_size=batch_size, shuffle=True)
val_loader_h   = DataLoader(TensorDataset(Xh_val_t, yh_val_t), batch_size=batch_size, shuffle=False)
train_loader_r = DataLoader(TensorDataset(Xr_train_t, yr_train_t), batch_size=batch_size, shuffle=True)
val_loader_r   = DataLoader(TensorDataset(Xr_val_t, yr_val_t), batch_size=batch_size, shuffle=False)

# ---------------------------------------------
# Define a simple neural network model
# ---------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------
# Training loop for a given model and dataset
# ---------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss_total += criterion(outputs, targets).item() * inputs.size(0)
        val_loss = val_loss_total / len(val_loader.dataset)
        val_losses.append(val_loss)
    return train_losses, val_losses

# ---------------------------------------------
# Train and evaluate height prediction model
# ---------------------------------------------
model_height = SimpleNet(input_size=num_pd, hidden_size=64)
train_losses_h, val_losses_h = train_model(model_height, train_loader_h, val_loader_h, num_epochs=100)

# Compute RMSE on test set for height model
model_height.eval()
with torch.no_grad():
    preds_h = model_height(Xh_test_t)
test_mse_h = nn.MSELoss()(preds_h, yh_test_t).item()
rmse_height = np.sqrt(test_mse_h) * 2.0  # scale back (target was H/2)
print(f"Height model - Test RMSE: {rmse_height:.4f} m")

# ---------------------------------------------
# Train and evaluate radius prediction model
# ---------------------------------------------
model_radius = SimpleNet(input_size=num_pd, hidden_size=64)
train_losses_r, val_losses_r = train_model(model_radius, train_loader_r, val_loader_r, num_epochs=100)

model_radius.eval()
with torch.no_grad():
    preds_r = model_radius(Xr_test_t)
test_mse_r = nn.MSELoss()(preds_r, yr_test_t).item()
rmse_radius = np.sqrt(test_mse_r) * 0.5  # scale back (target was R/0.5)
print(f"Radius model - Test RMSE: {rmse_radius:.4f} m")

# ---------------------------------------------
# Plot training and validation loss curves
# ---------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(train_losses_h, label="Train Loss")
plt.plot(val_losses_h, label="Val Loss")
plt.title("Height Model Loss vs. Epoch")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(train_losses_r, label="Train Loss")
plt.plot(val_losses_r, label="Val Loss")
plt.title("Radius Model Loss vs. Epoch")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
plt.show()
