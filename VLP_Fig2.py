import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_vlp_figure_2():
    # --- 1. THIẾT LẬP THAM SỐ (Dựa trên hình học bài báo) ---
    # Kích thước phòng (L x W x H)
    ROOM_L = 5.0
    ROOM_W = 5.0
    ROOM_H = 3.0
    
    # Tạo lưới điểm đo (Receiver Grid) trên sàn (z=0)
    grid_step = 0.1  # Độ phân giải 0.1m
    x = np.arange(-ROOM_L/2, ROOM_L/2 + grid_step, grid_step)
    y = np.arange(-ROOM_W/2, ROOM_W/2 + grid_step, grid_step)
    X, Y = np.meshgrid(x, y)
    Z_RX = 0.0
    
    # --- 2. CẤU HÌNH ĐÈN LED (TRANSMITTERS) ---
    # Vị trí: Trung điểm các đường chéo từ tâm đến góc phòng
    # Tọa độ: (+/- 1.25, +/- 1.25, 3.0)
    offset = 1.25
    z_tx = 3.0
    led_positions = ([offset, offset, z_tx],    # LED 1 (Góc phần tư 1)
        [-offset, offset, z_tx],   # LED 2 (Góc phần tư 2)
        [-offset, -offset, z_tx],  # LED 3 (Góc phần tư 3)
        [offset, -offset, z_tx]    # LED 4 (Góc phần tư 4)
    )
    
    # Công suất
    P_TOTAL = 2.0           # Tổng công suất hệ thống (Watts)
    P_LED = P_TOTAL / 4.0   # 0.5 W mỗi đèn
    
    # --- 3. THAM SỐ KÊNH QUANG HỌC ---
    # Để tái tạo hình dạng 4 đỉnh của Hình 2, ta dùng m = 12.5
    # (Nếu dùng m=1 theo bảng, hình sẽ ra dạng mái vòm đơn)
    m_order = 12.5          
    
    A_PD = 1e-4             # Diện tích PD (1 cm^2)
    Ts = 1.0                # Độ lợi bộ lọc
    n = 1.5                 # Chiết suất
    FOV_deg = 60.0          # Góc nhìn (độ)
    FOV_rad = np.deg2rad(FOV_deg)
    rho = 0.8               # Hệ số phản xạ tường
    
    # Độ lợi bộ tập trung quang (Concentrator Gain)
    g_conc = (n**2) / (np.sin(FOV_rad)**2)
    
    # Thành phần Phản xạ khuếch tán (Diffuse - Integrating Sphere)
    # H_diff = (rho * A) / (A_room * (1 - rho))
    area_room = 2 * (ROOM_L*ROOM_W + ROOM_L*ROOM_H + ROOM_W*ROOM_H)
    H_diffuse = (rho * A_PD) / (area_room * (1 - rho))
    
    # --- 4. TÍNH TOÁN CÔNG SUẤT THU ---
    # Khởi tạo ma trận công suất tổng (đơn vị Watt)
    P_total_watts = np.zeros_like(X)
    
    # Duyệt qua từng đèn LED để cộng dồn công suất
    for led in led_positions:
        lx, ly, lz = led
        
        # Tính khoảng cách từ đèn đến từng điểm trên lưới
        dist_sq = (X - lx)**2 + (Y - ly)**2 + (Z_RX - lz)**2
        dist = np.sqrt(dist_sq)
        
        # Tính Cosine góc phát (phi) và góc tới (psi)
        # Giả sử đèn hướng thẳng xuống và sàn phẳng: phi = psi
        h = lz - Z_RX
        cos_phi = h / dist
        cos_psi = cos_phi
        
        # Tính H_LoS (Line of Sight Channel Gain)
        # Công thức: H = [(m+1)A / 2pi*d^2] * cos^m(phi) * Ts * g * cos(psi)
        H_los = ((m_order + 1) * A_PD) / (2 * np.pi * dist_sq) * \
                (cos_phi ** m_order) * \
                Ts * g_conc * cos_psi
        
        # Kiểm tra điều kiện FOV (Field of View)
        # Góc tới thực tế (psi_angle)
        psi_angle = np.arccos(np.clip(cos_psi, -1.0, 1.0))
        # Tạo mặt nạ: điểm nào ngoài FOV thì Gain = 0
        mask_fov = psi_angle <= FOV_rad
        H_los = H_los * mask_fov
        
        # Tổng hợp kênh: LoS + Diffuse
        H_total = H_los + H_diffuse
        
        # Cộng công suất đèn hiện tại vào tổng
        P_total_watts += P_LED * H_total

    # --- 5. CHUYỂN ĐỔI SANG dBm ---
    # P(dBm) = 10 * log10( P(W) / 1mW )
    # Cộng thêm 1e-20 để tránh lỗi log(0) nếu có điểm chết
    P_total_dBm = 10 * np.log10((P_total_watts + 1e-20) * 1000)
    
    # --- 6. VẼ BIỂU ĐỒ 3D ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ bề mặt (Surface Plot)
    surf = ax.plot_surface(X, Y, P_total_dBm, cmap='jet', 
                           edgecolor='none', antialiased=True, alpha=0.9)
    
    # Thiết lập trục và nhãn
    ax.set_title(f'Mô phỏng Hình 2: Hồ sơ Công suất Thu (m={m_order})', fontsize=14)
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_zlabel('Received Power (dBm)')
    
    # Giới hạn trục theo bài báo
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    
    # Tinh chỉnh giới hạn trục Z để hiển thị rõ hình dạng
    z_min = np.min(P_total_dBm)
    z_max = np.max(P_total_dBm)
    ax.set_zlim(z_min, z_max + 2)
    
    # Thêm thanh màu (Colorbar)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label='Power (dBm)')
    
    # Điều chỉnh góc nhìn (View angle) cho giống hình minh họa
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_vlp_figure_2()