import numpy as np

# Tham số hệ thống
room_size = (5.0, 5.0, 3.0)  # (width, depth, height) in meters
LED_height = room_size[2]    # LED at ceiling (z = 3.0 m)
# Vị trí 4 LED: midpoint of diagonals from center to corners:contentReference[oaicite:16]{index=16}
LED_positions = np.array([
    [1.25, 1.25, LED_height],
    [1.25, 3.75, LED_height],
    [3.75, 1.25, LED_height],
    [3.75, 3.75, LED_height],
])
num_LED = LED_positions.shape[0]

# Tham số kênh quang
P_total = 2.0  # tổng công suất phát (W):contentReference[oaicite:17]{index=17}
P_LED = P_total / num_LED  # công suất mỗi LED (W) (các LED phát đều công suất):contentReference[oaicite:18]{index=18}
m = 1  # bậc Lambert (với semi-angle ~60°, m ≈ 1):contentReference[oaicite:19]{index=19}
A_PD = 1e-4  # diện tích PD (m^2):contentReference[oaicite:20]{index=20}
Ts = 1.0     # optical filter gain (giả sử 1)
n = 1.5      # chiết suất của thấu kính PD:contentReference[oaicite:21]{index=21}
psi_c = np.deg2rad(60.0)  # FOV 60° (đổi sang radian):contentReference[oaicite:22]{index=22}
# Độ lợi tập trung g(ψ) = n^2 / sin^2(psi_c), áp dụng trong giới hạn FOV
g_concentrator = (n**2) / (np.sin(psi_c)**2)

# Tạo lưới các điểm PD trên sàn (0<=x<=5, 0<=y<=5, z=0)
grid_step = 0.2  # khoảng cách giữa các PD lân cận (m):contentReference[oaicite:23]{index=23}:contentReference[oaicite:24]{index=24}
x_coords = np.arange(0.0, room_size[0] + 1e-9, grid_step)
y_coords = np.arange(0.0, room_size[1] + 1e-9, grid_step)
XX, YY = np.meshgrid(x_coords, y_coords)  # tạo lưới tọa độ
PD_positions = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(XX.size)))
num_PD = PD_positions.shape[0]  # tổng số PD

# Tiền tính các đại lượng khoảng cách và góc cho mỗi cặp LED-PD khi không có vật cản
# base_H0[led_index, pd_index] = H0 (hệ số kênh LoS) từ LED đó đến PD đó (nếu trong FOV, chưa tính P_LED)
base_H0 = np.zeros((num_LED, num_PD), dtype=float)
# Tọa độ LED và PD tách riêng để vector hóa tính toán
LED_xy = LED_positions[:, :2]    # shape (4,2)
PD_xy = PD_positions[:, :2].T    # shape (2, 676) for broadcasting
# Vector khoảng cách 3D từ LED tới PD
for li in range(num_LED):
    # vector từ LED li đến tất cả PD: (dx, dy, dz)
    dx = PD_xy[0] - LED_xy[li, 0]  # shape (num_PD,)
    dy = PD_xy[1] - LED_xy[li, 1]
    dz = 0.0 - LED_positions[li, 2]  # = -LED_height for all PD, since PD z=0
    dist2 = dx**2 + dy**2 + dz**2            # bình phương khoảng cách 3D
    D = np.sqrt(dist2)                      # khoảng cách 3D
    # Tính cos(phi) = cos(psi) = (độ chênh cao / khoảng cách 3D)
    cos_psi = abs(dz) / D                   # dz is negative, use absolute for angle
    # Kiểm tra xem PD có nằm trong FOV của LED không
    in_FOV = cos_psi >= np.cos(psi_c)       # True nếu góc tới <= 60°
    # Áp dụng công thức H0 cho các PD trong FOV
    # (m+1)*A/(2π D^2) * cos^m(phi) * cos(psi) * Ts * g(psi)
    # Với m=1: (m+1)*cos^m(phi)*cos(psi) = 2 * cos(phi)^1 * cos(psi) = 2 * cos_psi^2
    # Nên H0 = 2 * A * cos_psi^2 * Ts * g / (2π D^2) = A * cos_psi^2 * Ts * g / (π D^2)
    H0_vals = np.zeros_like(D)
    # Chỉ tính H0 cho PD trong FOV
    H0_vals[in_FOV] = (A_PD * (cos_psi[in_FOV]**2) * Ts * g_concentrator) / (np.pi * (D[in_FOV]**2))
    base_H0[li] = H0_vals

# Bây giờ base_H0 chứa hệ số kênh (LoS) không vật cản của từng LED tới từng PD.
# Công suất nhận không vật cản tại PD = P_LED * base_H0 (tổng cộng từ các LED).

# Hàm kiểm tra che khuất: xác định PD nào bị LED->PD bị chặn bởi vật
def compute_block_mask(LED_pos, PD_xy_coords, obj_xy, obj_h, obj_r):
    """
    Tính mảng bool cho biết với LED tại LED_pos, những PD (tương ứng PD_xy_coords 2xN) nào bị 
    chặn (True nếu bị chặn) bởi vật cản ở obj_xy (chiều cao obj_h, bán kính obj_r).
    """
    # Vector khoảng cách ngang LED->PD và LED->vật
    LED_x, LED_y, LED_z = LED_pos
    # PD_xy_coords: shape (2, N) (x row, y row for N PDs)
    dx = PD_xy_coords[0] - LED_x
    dy = PD_xy_coords[1] - LED_y
    # Khoảng cách ngang từ LED tới PD và LED tới vật
    d_T_sq = dx**2 + dy**2
    d_T = np.sqrt(d_T_sq)
    d_B = np.sqrt((obj_xy[0] - LED_x)**2 + (obj_xy[1] - LED_y)**2)
    # Điều kiện 2: h_B / d_B >= h_T / d_T  (với PD cao 0, LED cao h_T)
    # => h_B * d_T >= h_T * d_B. (Xét d_B > 0 để tránh chia 0)
    cond2 = np.full(dx.shape, False)
    if d_B > 0:
        cond2 = (obj_h * d_T) >= (LED_z * d_B)
    # Điều kiện 1: khoảng cách từ đường thẳng LED-PD tới tâm vật <= r.
    # Tính khoảng cách này qua vector pháp tuyến:
    # khoảng cách = |(LED->PD) x (LED->Obj)| / |LED->PD| (tích chéo 2D)
    # Ở 2D: cross_z = dx*(obj_y-LED_y) - dy*(obj_x-LED_x)
    cross_z = dx * (obj_xy[1] - LED_y) - dy * (obj_xy[0] - LED_x)
    dist_line = np.abs(cross_z) / d_T  # khoảng cách từ tâm vật tới đường thẳng nối LED-PD
    cond1 = dist_line <= obj_r
    # Ngoài ra, cần chắc chắn vật nằm giữa LED và PD (giao điểm vuông góc nằm trên đoạn thẳng):
    # Kiểm tra t (0<=t<=1) với t = (đoạn LED->vật proj trên LED->PD) / |LED->PD|^2
    t = ((obj_xy[0] - LED_x) * dx + (obj_xy[1] - LED_y) * dy) / d_T_sq
    between = (t >= 0.0) & (t <= 1.0)
    # Mặt khác, nếu LED nằm ngay trên vật (d_B rất nhỏ) thì chỉ xét PD xa hơn vật (t>1)
    # Trường hợp này hiếm và có thể bỏ qua hoặc xử lý riêng nếu cần.
    # Kết hợp điều kiện:
    blocked = cond1 & cond2 & between
    return blocked

# Số mẫu cần sinh
num_samples = 1000

# Khởi tạo mảng dữ liệu
X_height = np.zeros((num_samples, num_PD), dtype=float)
y_height = np.zeros(num_samples, dtype=float)
X_radius = np.zeros((num_samples, num_PD), dtype=float)
y_radius = np.zeros(num_samples, dtype=float)

# Giá trị cố định mặc định (nếu muốn cố định tham số không phải đích)
default_h = 1.0   # chiều cao mặc định (m)
default_r = 0.1   # bán kính mặc định (m)

# Sinh dữ liệu mẫu
for i in range(num_samples):
    # Sinh ngẫu nhiên chiều cao và bán kính theo phân bố đều:contentReference[oaicite:25]{index=25}
    h = np.random.uniform(0.0, 2.0)    # chiều cao 0–2 m
    r = np.random.uniform(0.0, 0.5)   # bán kính 0–0.5 m
    # Tọa độ (x, y) ngẫu nhiên cho tâm vật, đảm bảo vật nằm trọn trong phòng
    x_obj = np.random.uniform(r, room_size[0] - r)
    y_obj = np.random.uniform(r, room_size[1] - r)
    obj_xy = (x_obj, y_obj)
    # Tính profile công suất thu P tại tất cả PD cho mẫu này
    total_power = np.zeros(num_PD, dtype=float)
    # Tính đóng góp từ mỗi LED
    for li in range(num_LED):
        # Áp dụng che khuất
        blocked_mask = compute_block_mask(LED_positions[li], PD_xy, obj_xy, h, r)
        # Công suất từ LED li tới các PD (nếu không bị chặn)
        # P_LED * H0 cho từng PD
        P_contrib = P_LED * base_H0[li]  # vector độ dài num_PD
        # Loại bỏ những PD bị chắn (blocked_mask True -> đặt 0)
        P_contrib[blocked_mask] = 0.0
        # Cộng vào tổng công suất tại PD
        total_power += P_contrib
    # Lưu kết quả đầu vào (đã flatten) và nhãn cho từng task
    X_height[i, :] = total_power  # vector hóa ma trận công suất
    y_height[i] = h               # nhãn chiều cao
    X_radius[i, :] = total_power
    y_radius[i] = r               # nhãn bán kính

# Lưu dữ liệu ra file .npz
np.savez("vlp_height_data.npz", X=X_height, y=y_height)
np.savez("vlp_radius_data.npz", X=X_radius, y=y_radius)
print("Generated {} samples for height estimation and {} samples for radius estimation.".format(num_samples, num_samples))
