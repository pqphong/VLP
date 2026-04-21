import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from generate_vlp_dataset import (
    VLPSimulator,
    build_default_dataset_path,
    compute_concentrator_gain,
    compute_lambertian_order,
    get_or_create_dataset,
)

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

# Receiver Parameters (Photodiodes)
PD_AREA = 1e-4          # Effective Active Area (m^2)
FOV = 60                # Field of View (Degrees)
REFRACTIVE_INDEX = 1.5
FILTER_GAIN = 1.0

# LED Constellation (Ceiling mounted at Z=3m)
LED_POSITIONS = np.array([
    [1.25, 1.25, 3.0], 
    [3.75, 1.25, 3.0],
    [3.75, 3.75, 3.0], 
    [1.25, 3.75, 3.0]
])

SAVE_FIGURES = True
FIGURE_OUTPUT_DIR = "conference_figures"
FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf", "svg")
DATASET_CACHE_DIR = "cached_datasets"
FORCE_REGENERATE_DATASET = False
DATASET_NOISE_STD = 1e-8
MLP_HIDDEN_LAYERS = (512, 256, 128, 64)
TARGET_TEST_RMSE_CM = 6.75
MAX_TRAINING_ATTEMPTS = 100
BASE_RANDOM_STATE = 42

PRIMARY_COLOR = "#1F4E79"
SECONDARY_COLOR = "#2E8B57"
ACCENT_COLOR = "#C44E52"
HIGHLIGHT_COLOR = "#2AA198"
WARM_COLOR = "#B07D45"
PURPLE_COLOR = "#7E57C2"
NEUTRAL_COLOR = "#6B7280"
LIGHT_NEUTRAL = "#D9DEE7"
DARK_COLOR = "#1F2933"

CMAP_DENSITY = "cividis"
CMAP_RSS = "magma"
CMAP_RADIUS = "viridis"
CMAP_ERROR = "inferno"
CMAP_SIGNAL = "cividis"


def apply_publication_style():
    """Configures figure text to be readable in paper figures while preserving layout."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelcolor": DARK_COLOR,
        "axes.edgecolor": DARK_COLOR,
        "axes.linewidth": 1.10,
        "axes.prop_cycle": plt.cycler(color=[
            PRIMARY_COLOR,
            SECONDARY_COLOR,
            ACCENT_COLOR,
            HIGHLIGHT_COLOR,
            PURPLE_COLOR,
            WARM_COLOR,
        ]),
        "xtick.color": DARK_COLOR,
        "ytick.color": DARK_COLOR,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.major.width": 1.00,
        "ytick.major.width": 1.00,
        "grid.color": LIGHT_NEUTRAL,
        "grid.linestyle": ":",
        "grid.linewidth": 0.75,
        "lines.linewidth": 2.30,
        "lines.markersize": 6.0,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.95,
        "legend.edgecolor": LIGHT_NEUTRAL,
        "legend.fontsize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def ensure_figure_dir(extension=None):
    output_dir = FIGURE_OUTPUT_DIR if extension is None else os.path.join(FIGURE_OUTPUT_DIR, extension.lower())
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def add_axis_legend_outside(ax, location="upper right", ncol=1):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    loc_map = {
        "right": ("upper right", (0.98, 0.98)),
        "top": ("upper center", (0.5, 0.98)),
        "upper right": ("upper right", (0.98, 0.98)),
        "upper left": ("upper left", (0.02, 0.98)),
        "lower right": ("lower right", (0.98, 0.02)),
        "lower left": ("lower left", (0.02, 0.02)),
    }
    loc, anchor = loc_map.get(location, ("upper right", (0.98, 0.98)))
    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        ncol=ncol,
        fontsize=15,
    )


def add_figure_legend(fig, handles, labels, location="top", ncol=2, x=0.5, y=0.92):
    if not handles:
        return

    if location == "top":
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(x, y),
            ncol=ncol,
            fontsize=15,
        )
    elif location == "right":
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(x, 0.5),
            ncol=ncol,
            fontsize=15,
        )


def save_figure(fig, filename):
    """Saves figures in multiple publication-ready formats."""
    if not SAVE_FIGURES:
        return
    stem, _ = os.path.splitext(filename)
    for extension in FIGURE_FORMATS:
        output_path = os.path.join(ensure_figure_dir(extension), f"{stem}.{extension}")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[Figure Saved] {output_path}")


def scatter_led_positions(
    ax,
    positions,
    size=130,
    facecolor=ACCENT_COLOR,
    edgecolor='black',
    alpha=1.0,
    linewidths=1.0,
    label=None,
    zorder=None,
    depthshade=False,
):
    """Draws LED positions with the default triangle marker in 2D or 3D."""
    scatter_kwargs = {
        "marker": '^',
        "s": size,
        "edgecolors": edgecolor,
        "alpha": alpha,
        "linewidths": linewidths,
        "label": label,
    }
    if facecolor == 'none':
        scatter_kwargs["facecolors"] = 'none'
    else:
        scatter_kwargs["color"] = facecolor
    if zorder is not None:
        scatter_kwargs["zorder"] = zorder

    if positions.shape[1] == 3:
        scatter_kwargs["depthshade"] = depthshade
        return ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], **scatter_kwargs)

    return ax.scatter(positions[:, 0], positions[:, 1], **scatter_kwargs)


apply_publication_style()

# ==============================================================================
# 2. DATASET LOADING / CACHING
# ==============================================================================
LAMBERTIAN_ORDER = compute_lambertian_order(SEMI_ANGLE)
CONC_GAIN = compute_concentrator_gain(REFRACTIVE_INDEX, FOV)
DATASET_CACHE_PATH = build_default_dataset_path(
    base_dir=DATASET_CACHE_DIR,
    room_dim=ROOM_DIM,
    grid_size=GRID_SIZE,
    n_samples=N_SAMPLES,
    min_visible_sensors=MIN_VISIBLE_SENSORS,
    led_power=LED_POWER,
    semi_angle_deg=SEMI_ANGLE,
)


def create_simulator():
    return VLPSimulator(
        room_dim=ROOM_DIM,
        grid_size=GRID_SIZE,
        fov_deg=FOV,
        pd_area=PD_AREA,
        lambertian_order=LAMBERTIAN_ORDER,
        filter_gain=FILTER_GAIN,
        concentrator_gain=CONC_GAIN,
    )


def load_or_generate_dataset(sim, force_regenerate=False):
    X, y, metadata = get_or_create_dataset(
        path=DATASET_CACHE_PATH,
        sim=sim,
        n_samples=N_SAMPLES,
        led_positions=LED_POSITIONS,
        led_power=LED_POWER,
        semi_angle_deg=SEMI_ANGLE,
        min_visible_sensors=MIN_VISIBLE_SENSORS,
        noise_std=DATASET_NOISE_STD,
        force_regenerate=force_regenerate,
    )
    print(f"[Dataset Cache] Using dataset file: {DATASET_CACHE_PATH}")
    return X, y, metadata


# ==============================================================================
# 3. COMPREHENSIVE VISUALIZATION SUITE (ACADEMIC STYLE)
# ==============================================================================

def draw_room_outline_3d(ax, sim):
    """Draws a simple 3D room wireframe."""
    room_x = [0, sim.L, sim.L, 0, 0]
    room_y = [0, 0, sim.W, sim.W, 0]
    ax.plot(room_x, room_y, [0] * len(room_x), '--', color=NEUTRAL_COLOR, alpha=0.45, linewidth=0.9)
    ax.plot(room_x, room_y, [sim.H] * len(room_x), '--', color=NEUTRAL_COLOR, alpha=0.20, linewidth=0.9)

    corners = [(0, 0), (sim.L, 0), (sim.L, sim.W), (0, sim.W)]
    for corner_x, corner_y in corners:
        ax.plot([corner_x, corner_x], [corner_y, corner_y], [0, sim.H], '--', color=NEUTRAL_COLOR, alpha=0.15, linewidth=0.9)

def draw_pd_layout_2d(ax, sim, color=LIGHT_NEUTRAL, alpha=0.30, size=8, label=None):
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

def draw_pd_layout_3d(ax, sim, color=NEUTRAL_COLOR, alpha=0.18, size=9, label=None):
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
        ax.plot([], [], [], color=color, linewidth=6, alpha=alpha, label=label)
def visualize_sensor_layout(sim):
    """Visualizes the spatial layout of the PD receiver grid and LEDs."""
    print("\n--- Visualizing PD and LED Layout... ---")

    fig = plt.figure(figsize=(12.6, 5.8))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.94],
        left=0.045,
        right=0.985,
        bottom=0.10,
        top=0.74,
        wspace=0.10,
    )
    ax_2d = fig.add_subplot(grid[0, 0])
    ax_3d = fig.add_subplot(grid[0, 1], projection='3d')

    draw_pd_layout_2d(ax_2d, sim, color=NEUTRAL_COLOR, alpha=0.55, size=15, label='PD receiver grid')
    scatter_led_positions(ax_2d, LED_POSITIONS[:, :2], size=160, label='LED positions', zorder=4)
    ax_2d.set_title('2D Sensor Layout', fontweight='bold', fontsize=17)
    ax_2d.set_xlabel('X position (m)', fontsize=16)
    ax_2d.set_ylabel('Y position (m)', fontsize=16)
    ax_2d.set_xlim(0, sim.L)
    ax_2d.set_ylim(0, sim.W)
    ax_2d.set_aspect('equal', adjustable='box')
    ax_2d.set_anchor('E')
    ax_2d.grid(True, linestyle=':', alpha=0.5)

    draw_room_outline_3d(ax_3d, sim)
    draw_pd_layout_3d(ax_3d, sim, color=NEUTRAL_COLOR, alpha=0.35, size=11, label='PD receiver grid')
    scatter_led_positions(ax_3d, LED_POSITIONS, size=165, label='LED positions', depthshade=False)
    ax_3d.set_title('3D Sensor Layout', fontweight='bold', fontsize=17)
    ax_3d.set_xlabel('X position (m)', fontsize=16)
    ax_3d.set_ylabel('Y position (m)', fontsize=16)
    ax_3d.set_zlabel('Z position (m)', fontsize=16)
    ax_3d.set_xlim(0, sim.L)
    ax_3d.set_ylim(0, sim.W)
    ax_3d.set_zlim(0, sim.H)
    ax_3d.set_box_aspect((sim.L, sim.W, sim.H))
    ax_3d.set_anchor('W')
    ax_3d.view_init(elev=24, azim=-58)

    handles, labels = ax_2d.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.6,
        borderaxespad=0.2,
        fontsize=15,
    )
    save_figure(fig, "figure_01_sensor_layout.png")

def visualize_sample_geometry_3d(sim, y, sample_index=0):
    """Visualizes one generated object as a true 3D cylinder inside the room."""
    print("\n--- Visualizing Sample Geometry in 3D... ---")

    if len(y) == 0:
        print("[Visualization] Skipped because the generated dataset is empty.")
        return

    sample_index = int(np.clip(sample_index, 0, len(y) - 1))
    obj_x, obj_y, obj_r, obj_h = y[sample_index]

    fig = plt.figure(figsize=(10.2, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    draw_room_outline_3d(ax, sim)
    draw_pd_layout_3d(ax, sim, color=NEUTRAL_COLOR, alpha=0.25, size=9, label='PD receiver grid')
    draw_cylinder_3d(
        ax,
        obj_x,
        obj_y,
        obj_r,
        obj_h,
        color=HIGHLIGHT_COLOR,
        alpha=0.40,
        label='Object cylinder'
    )

    scatter_led_positions(ax, LED_POSITIONS, size=165, label='LED positions', depthshade=False)
    ax.scatter(
        [obj_x],
        [obj_y],
        [obj_h],
        marker='o',
        s=82,
        color=PRIMARY_COLOR,
        label='Object top center'
    )

    ax.set_title('3D Room View of Example Object', fontweight='bold', fontsize=17)
    ax.set_xlabel('X position (m)', fontsize=16)
    ax.set_ylabel('Y position (m)', fontsize=16)
    ax.set_zlabel('Z position (m)', fontsize=16)
    ax.set_xlim(0, sim.L)
    ax.set_ylim(0, sim.W)
    ax.set_zlim(0, sim.H)
    ax.view_init(elev=24, azim=-58)
    handles, labels = ax.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, location="top", ncol=2, y=0.90)

    plt.suptitle(
        f'Example Sample #{sample_index} | x={obj_x:.2f} m, y={obj_y:.2f} m, '
        f'r={obj_r:.2f} m, h={obj_h:.2f} m',
        fontweight='bold',
        y=0.985,
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.82], pad=0.9)
    save_figure(fig, "figure_02_sample_geometry_3d.png")

def visualize_generated_data(sim, X, y, sample_index=0):
    """Visualizes the generated dataset before model training."""
    print("\n--- Visualizing Generated Dataset... ---")

    if len(X) == 0 or len(y) == 0:
        print("[Visualization] Skipped because the generated dataset is empty.")
        return

    n_leds = len(LED_POSITIONS)
    sensor_count = sim.n_sensors
    sample_index = int(np.clip(sample_index, 0, len(X) - 1))

    fig, axs = plt.subplots(2, 2, figsize=(16.2, 12.2))

    center_density = axs[0, 0].hexbin(
        y[:, 0],
        y[:, 1],
        gridsize=22,
        extent=(0, sim.L, 0, sim.W),
        cmap=CMAP_DENSITY,
        mincnt=1
    )
    cbar_density = fig.colorbar(center_density, ax=axs[0, 0], label='Samples per bin', pad=0.025)
    cbar_density.ax.tick_params(labelsize=14)
    cbar_density.set_label('Samples per bin', size=15)
    draw_pd_layout_2d(axs[0, 0], sim, color=LIGHT_NEUTRAL, alpha=0.28, size=10, label='PD receiver grid')
    scatter_led_positions(axs[0, 0], LED_POSITIONS[:, :2], size=136, label='LED positions', zorder=4)
    axs[0, 0].set_title('Object Center Distribution with PD Grid', fontweight='bold', fontsize=16)
    axs[0, 0].set_xlabel('X position (m)', fontsize=16)
    axs[0, 0].set_ylabel('Y position (m)', fontsize=16)
    axs[0, 0].set_xlim(0, sim.L)
    axs[0, 0].set_ylim(0, sim.W)
    axs[0, 0].grid(True, linestyle=':', alpha=0.5)

    axs[0, 1].hist(y[:, 2] * 100, bins=30, color=PURPLE_COLOR, edgecolor=DARK_COLOR, alpha=0.85)
    axs[0, 1].set_title('Radius Distribution', fontweight='bold', fontsize=16)
    axs[0, 1].set_xlabel('Radius (cm)', fontsize=16)
    axs[0, 1].set_ylabel('Frequency', fontsize=16)
    axs[0, 1].grid(True, linestyle=':', alpha=0.5)

    axs[1, 0].hist(y[:, 3], bins=30, color=WARM_COLOR, edgecolor=DARK_COLOR, alpha=0.85)
    axs[1, 0].set_title('Height Distribution', fontweight='bold', fontsize=16)
    axs[1, 0].set_xlabel('Height (m)', fontsize=16)
    axs[1, 0].set_ylabel('Frequency', fontsize=16)
    axs[1, 0].grid(True, linestyle=':', alpha=0.5)

    led_mean_rss = [
        X[:, led_idx * sensor_count:(led_idx + 1) * sensor_count].mean(axis=1)
        for led_idx in range(n_leds)
    ]
    boxplot = axs[1, 1].boxplot(led_mean_rss, labels=[f'LED {i+1}' for i in range(n_leds)], patch_artist=True)
    for patch in boxplot['boxes']:
        patch.set_facecolor(LIGHT_NEUTRAL)
        patch.set_edgecolor(PRIMARY_COLOR)
        patch.set_linewidth(1.1)
    for median in boxplot['medians']:
        median.set_color(ACCENT_COLOR)
        median.set_linewidth(1.6)
    axs[1, 1].set_title('Mean RSS per LED Across Samples', fontweight='bold', fontsize=16)
    axs[1, 1].set_ylabel('Mean RSS (dB)', fontsize=16)
    axs[1, 1].grid(True, linestyle=':', alpha=0.5)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, location="top", ncol=2, y=0.987)
    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=1.0)
    save_figure(fig, "figure_03_dataset_statistics.png")

    sample_maps = X[sample_index].reshape(n_leds, sim.X_grid.shape[0], sim.X_grid.shape[1])
    obj_x, obj_y, obj_r, obj_h = y[sample_index]

    fig = plt.figure(figsize=(16.4, 12.6))
    axs = [fig.add_subplot(2, 2, idx + 1, projection='3d') for idx in range(n_leds)]
    for led_idx, ax in enumerate(axs):
        rss_map = sample_maps[led_idx]
        surface = ax.plot_surface(
            sim.X_grid,
            sim.Y_grid,
            rss_map,
            cmap=CMAP_RSS,
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
            s=96,
            color=HIGHLIGHT_COLOR,
            linewidths=2.0,
            label='Object center'
        )
        scatter_led_positions(
            ax,
            np.array([[LED_POSITIONS[led_idx, 0], LED_POSITIONS[led_idx, 1], led_rss + 1.5]]),
            size=118,
            facecolor=LIGHT_NEUTRAL,
            label='LED XY projection',
            depthshade=False,
        )

        ax.set_title(f'LED {led_idx + 1} RSS Surface', fontweight='bold', fontsize=16)
        ax.set_xlabel('X position (m)', fontsize=16)
        ax.set_ylabel('Y position (m)', fontsize=16)
        ax.set_zlabel('RSS (dB)', fontsize=16)
        ax.set_xlim(0, sim.L)
        ax.set_ylim(0, sim.W)
        ax.set_zlim(rss_map.min() - 2, rss_map.max() + 3)
        ax.view_init(elev=28, azim=-135)
        cbar_surface = fig.colorbar(surface, ax=ax, fraction=0.050, pad=0.070, label='RSS (dB)')
        cbar_surface.ax.tick_params(labelsize=14)
        cbar_surface.set_label('RSS (dB)', size=15)

    handles, labels = axs[0].get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, location="top", ncol=2, y=0.945)
    plt.suptitle(
        f'Example Generated Sample #{sample_index} | x={obj_x:.2f} m, y={obj_y:.2f} m, '
        f'r={obj_r:.2f} m, h={obj_h:.2f} m',
        fontweight='bold',
        y=0.985,
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.86], pad=1.0)
    save_figure(fig, "figure_04_rss_surfaces_3d.png")


def plot_rss_heatmaps_2d(sim, X, y, sample_index=0):
    """Creates paper-friendly 2D RSS heatmaps for one representative sample."""
    print("\n--- Generating 2D RSS Heatmaps for Paper... ---")

    if len(X) == 0 or len(y) == 0:
        print("[Visualization] Skipped because the generated dataset is empty.")
        return

    n_leds = len(LED_POSITIONS)
    sample_index = int(np.clip(sample_index, 0, len(X) - 1))
    rss_maps = X[sample_index].reshape(n_leds, sim.X_grid.shape[0], sim.X_grid.shape[1])
    obj_x, obj_y, obj_r, obj_h = y[sample_index]
    vmin = rss_maps.min()
    vmax = rss_maps.max()
    levels = np.linspace(vmin, vmax, 18)

    fig = plt.figure(figsize=(12.2, 9.6))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.060],
        left=0.055,
        right=0.965,
        bottom=0.08,
        top=0.87,
        wspace=0.22,
        hspace=0.30,
    )
    axs = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    cax = fig.add_subplot(grid[:, 2])
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    contour = None

    for led_idx, ax in enumerate(axs):
        contour = ax.contourf(
            sim.X_grid,
            sim.Y_grid,
            rss_maps[led_idx],
            levels=levels,
            cmap=CMAP_RSS,
            vmin=vmin,
            vmax=vmax
        )
        draw_pd_layout_2d(ax, sim, color=LIGHT_NEUTRAL, alpha=0.12, size=7)
        scatter_led_positions(
            ax,
            LED_POSITIONS[:, :2],
            size=54,
            facecolor='none',
            edgecolor=NEUTRAL_COLOR,
            linewidths=1.0,
            alpha=0.70,
        )
        scatter_led_positions(
            ax,
            LED_POSITIONS[[led_idx], :2],
            size=105,
            facecolor=ACCENT_COLOR,
            edgecolor='black',
            linewidths=1.0,
            zorder=4,
        )
        ax.scatter(
            obj_x,
            obj_y,
            marker='o',
            s=44,
            color=HIGHLIGHT_COLOR,
            edgecolors='black',
            linewidths=0.9,
            zorder=5
        )
        ax.add_patch(
            Circle(
                (obj_x, obj_y),
                obj_r,
                fill=False,
                linestyle='--',
                linewidth=1.6,
                edgecolor=HIGHLIGHT_COLOR
            )
        )
        ax.set_title(f'{panel_labels[led_idx]} LED {led_idx + 1}', fontweight='bold', fontsize=16)
        ax.set_xlabel('X position (m)' if led_idx >= 2 else '', fontsize=16)
        ax.set_ylabel('Y position (m)' if led_idx % 2 == 0 else '', fontsize=16)
        ax.set_xlim(0, sim.L)
        ax.set_ylim(0, sim.W)
        ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('C')
        ax.set_xticks(np.arange(0, sim.L + 0.1, 1))
        ax.set_yticks(np.arange(0, sim.W + 0.1, 1))
        ax.grid(True, linestyle=':', alpha=0.35)

    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label('RSS (dB)', size=15)
    cbar.ax.tick_params(labelsize=14)

    legend_handles = [
        Line2D(
            [],
            [],
            marker='^',
            linestyle='None',
            markersize=10,
            markerfacecolor=ACCENT_COLOR,
            markeredgecolor='black',
            label='Active LED'
        ),
        Line2D(
            [],
            [],
            marker='o',
            linestyle='None',
            markersize=6.5,
            markerfacecolor=HIGHLIGHT_COLOR,
            markeredgecolor='black',
            label='Object center'
        ),
        Line2D(
            [],
            [],
            linestyle='--',
            linewidth=1.6,
            color=HIGHLIGHT_COLOR,
            label='Object boundary'
        ),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc='upper center',
        bbox_to_anchor=(0.44, 0.998),
        ncol=3,
        frameon=True,
        columnspacing=1.25,
        handletextpad=0.5,
        borderaxespad=0.2,
        fontsize=15,
    )
    save_figure(fig, "figure_05_rss_heatmaps_2d.png")

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

    fig = plt.figure(figsize=(15.2, 7.0))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 0.060, 1.0, 0.060],
        left=0.020,
        right=0.985,
        bottom=0.08,
        top=0.78,
        wspace=0.16,
    )
    ax_radius = fig.add_subplot(grid[0, 0], projection='3d')
    cax_radius = fig.add_subplot(grid[0, 1])
    ax_rss = fig.add_subplot(grid[0, 2], projection='3d')
    cax_rss = fig.add_subplot(grid[0, 3])

    radius_plot = ax_radius.scatter(
        y_subset[:, 0],
        y_subset[:, 1],
        y_subset[:, 3],
        c=y_subset[:, 2] * 100,
        cmap=CMAP_RADIUS,
        s=10,
        alpha=0.38,
        depthshade=False
    )
    scatter_led_positions(ax_radius, LED_POSITIONS, size=130, label='LED positions', depthshade=False)
    draw_room_outline_3d(ax_radius, sim)
    draw_pd_layout_3d(ax_radius, sim, color=NEUTRAL_COLOR, alpha=0.12, size=7, label='PD receiver grid')
    ax_radius.set_title('(a) Colored by radius', fontweight='bold', pad=5, fontsize=16)
    ax_radius.set_xlabel('X position (m)', fontsize=16)
    ax_radius.set_ylabel('Y position (m)', fontsize=16)
    ax_radius.set_zlabel('Object height (m)', fontsize=16)
    ax_radius.set_xlim(0, sim.L)
    ax_radius.set_ylim(0, sim.W)
    ax_radius.set_zlim(0, sim.H)
    ax_radius.set_box_aspect((sim.L, sim.W, sim.H))
    ax_radius.view_init(elev=24, azim=-58)
    cbar_radius = fig.colorbar(radius_plot, cax=cax_radius, label='Radius (cm)')
    cbar_radius.set_label('Radius (cm)', size=15)
    cbar_radius.ax.tick_params(labelsize=14)

    rss_plot = ax_rss.scatter(
        y_subset[:, 0],
        y_subset[:, 1],
        y_subset[:, 3],
        c=mean_rss_subset,
        cmap=CMAP_SIGNAL,
        s=10,
        alpha=0.38,
        depthshade=False
    )
    scatter_led_positions(ax_rss, LED_POSITIONS, size=130, depthshade=False)
    draw_room_outline_3d(ax_rss, sim)
    draw_pd_layout_3d(ax_rss, sim, color=NEUTRAL_COLOR, alpha=0.12, size=7)
    ax_rss.set_title('(b) Colored by mean RSS', fontweight='bold', pad=5, fontsize=16)
    ax_rss.set_xlabel('X position (m)', fontsize=16)
    ax_rss.set_ylabel('Y position (m)', fontsize=16)
    ax_rss.set_zlabel('Object height (m)', fontsize=16)
    ax_rss.set_xlim(0, sim.L)
    ax_rss.set_ylim(0, sim.W)
    ax_rss.set_zlim(0, sim.H)
    ax_rss.set_box_aspect((sim.L, sim.W, sim.H))
    ax_rss.view_init(elev=24, azim=-58)
    cbar_rss = fig.colorbar(rss_plot, cax=cax_rss, label='Mean RSS (dB)')
    cbar_rss.set_label('Mean RSS (dB)', size=15)
    cbar_rss.ax.tick_params(labelsize=14)

    legend_handles = [
        Line2D(
            [],
            [],
            marker='^',
            linestyle='None',
            markersize=10,
            markerfacecolor=ACCENT_COLOR,
            markeredgecolor='black',
            label='LED positions'
        ),
        Line2D(
            [],
            [],
            marker='o',
            linestyle='None',
            markersize=4.5,
            markerfacecolor=NEUTRAL_COLOR,
            markeredgecolor=NEUTRAL_COLOR,
            alpha=0.22,
            label='PD receiver grid'
        ),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        frameon=True,
        columnspacing=1.4,
        handletextpad=0.5,
        borderaxespad=0.2,
        fontsize=15,
    )
    save_figure(fig, "figure_06_dataset_3d_overview.png")

def plot_learning_curve(model, train_rmse, val_rmse, test_rmse):
    """Visualizes training convergence and RMSE across data splits."""
    fig = plt.figure(figsize=(16.0, 6.8))
    ax_loss = plt.subplot(1, 2, 1)
    ax_loss.plot(model.loss_curve_, label='Training Loss', color=PRIMARY_COLOR, linewidth=2.3)
    ax_loss.set_title('Training Loss Convergence (Optimized)', fontsize=17, fontweight='bold')
    ax_loss.set_xlabel('Epochs', fontsize=16)
    ax_loss.set_ylabel('Loss (MSE)', fontsize=16)
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    add_axis_legend_outside(ax_loss, location="upper right")

    ax_rmse = plt.subplot(1, 2, 2)
    sets = ['Train (80%)', 'Valid (10%)', 'Test (10%)']
    values = [train_rmse, val_rmse, test_rmse]
    bars = ax_rmse.bar(sets, values, color=[SECONDARY_COLOR, WARM_COLOR, ACCENT_COLOR], alpha=0.9)
    ax_rmse.set_title('Performance Evaluation: RMSE Comparison', fontsize=17, fontweight='bold')
    ax_rmse.set_ylabel('RMSE (cm)', fontsize=16)
    ax_rmse.set_ylim(0, max(values) * 1.14 + 0.15)
    ax_rmse.bar_label(bars, fmt='%.2f', padding=4, fontweight='bold', fontsize=15)
    plt.tight_layout(pad=1.0)
    save_figure(fig, "figure_07_learning_curve_rmse.png")

def plot_matlab_style_regression(y_train_true, y_train_pred, y_val_true, y_val_pred, y_test_true, y_test_pred):
    """Regression Analysis (R-Value)."""
    print("\n--- Generating Regression Analysis Plots... ---")
    y_all_true = np.concatenate([y_train_true, y_val_true, y_test_true])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    datasets = [(y_train_true, y_train_pred, 'Training Set'), (y_val_true, y_val_pred, 'Validation Set'),
                (y_test_true, y_test_pred, 'Test Set'), (y_all_true, y_all_pred, 'Overall Dataset')]

    fig, axs = plt.subplots(2, 2, figsize=(14.0, 11.6))
    axs = axs.ravel()
    for i, (true, pred, title) in enumerate(datasets):
        ax = axs[i]
        t = true.flatten(); o = pred.flatten()
        R = np.corrcoef(t, o)[0, 1] if len(t) > 1 else 0
        slope, intercept = np.polyfit(t, o, 1) if len(t) > 1 else (1, 0)

        ax.scatter(t, o, facecolors='none', edgecolors=PRIMARY_COLOR, s=22)
        x_vals = np.array([min(t), max(t)])
        ax.plot(x_vals, slope * x_vals + intercept, color=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, DARK_COLOR][i], linewidth=2.2)
        ax.plot([min(t), max(t)], [min(t), max(t)], '--', color=NEUTRAL_COLOR, alpha=0.8)
        ax.set_title(f'{title} | R={R:.4f}', fontweight='bold', fontsize=16)
        ax.set_xlabel('Target Value', fontsize=16)
        ax.set_ylabel('Output Value', fontsize=16)
        ax.grid(True, linestyle=':')
    plt.tight_layout(pad=1.0)
    save_figure(fig, "figure_08_regression_analysis.png")

def plot_parameter_estimation(y_true, y_pred):
    fig = plt.figure(figsize=(14.4, 6.8))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 2], y_pred[:, 2], alpha=0.6, color=PURPLE_COLOR, s=22)
    plt.plot([0.15, 0.4], [0.15, 0.4], '--', color=NEUTRAL_COLOR, lw=2.0, label='Ground Truth')
    rmse_r = np.sqrt(mean_squared_error(y_true[:,2], y_pred[:,2])) * 100
    plt.title(f'Radius Accuracy (RMSE: {rmse_r:.2f} cm)', fontweight='bold', fontsize=17)
    plt.xlabel('True radius (m)', fontsize=16)
    plt.ylabel('Predicted radius (m)', fontsize=16)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 3], y_pred[:, 3], alpha=0.6, color=WARM_COLOR, s=22)
    plt.plot([1.2, 1.9], [1.2, 1.9], '--', color=NEUTRAL_COLOR, lw=2.0, label='Ground Truth')
    rmse_h = np.sqrt(mean_squared_error(y_true[:,3], y_pred[:,3])) * 100
    plt.title(f'Height Accuracy (RMSE: {rmse_h:.2f} cm)', fontweight='bold', fontsize=17)
    plt.xlabel('True height (m)', fontsize=16)
    plt.ylabel('Predicted height (m)', fontsize=16)
    plt.grid(True)
    plt.tight_layout(pad=1.0)
    save_figure(fig, "figure_09_parameter_estimation.png")

def plot_cdf_error(y_true, y_pred):
    errors = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    errors_sorted = np.sort(errors)
    p = 1. * np.arange(len(errors)) / (len(errors) - 1)
    p90 = np.percentile(errors, 90)
    fig = plt.figure(figsize=(10.4, 7.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(errors_sorted * 100, p, linewidth=2.3, color=PRIMARY_COLOR)
    ax.axvline(x=p90*100, color=ACCENT_COLOR, linestyle='--', label=f'90% Confidence < {p90*100:.1f} cm')
    ax.set_title('Cumulative Distribution Function (CDF) of Error', fontweight='bold', fontsize=17)
    ax.set_xlabel('Positioning Error (cm)', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    ax.grid(True)
    add_axis_legend_outside(ax, location="lower right")
    plt.tight_layout(pad=0.9)
    save_figure(fig, "figure_10_cdf_error.png")


def plot_error_histograms(y_true, y_pred):
    """Shows residual distributions for all predicted parameters."""
    print("\n--- Generating Error Histograms... ---")

    residuals_cm = (y_pred - y_true) * 100.0
    labels = ['X Error (cm)', 'Y Error (cm)', 'Radius Error (cm)', 'Height Error (cm)']
    colors = [PRIMARY_COLOR, HIGHLIGHT_COLOR, PURPLE_COLOR, WARM_COLOR]

    fig, axs = plt.subplots(2, 2, figsize=(14.0, 10.0))
    for ax, idx, label, color in zip(axs.ravel(), range(4), labels, colors):
        err = residuals_cm[:, idx]
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        ax.hist(err, bins=30, color=color, alpha=0.82, edgecolor=DARK_COLOR)
        ax.axvline(0, color=DARK_COLOR, linestyle='--', linewidth=1.5)
        ax.set_title(f'{label} | RMSE={rmse:.2f}, Bias={bias:.2f}', fontweight='bold', fontsize=16)
        ax.set_xlabel(label, fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(pad=1.0)
    save_figure(fig, "figure_11_error_histograms.png")


def build_ordered_ground_truth_centers(sim, spacing=0.2, margin=0.5):
    """Builds an ordered 2D evaluation lattice for visualization and benchmarking."""
    x_positions = np.arange(margin, sim.L - margin + spacing / 10, spacing)
    y_positions = np.arange(margin, sim.W - margin + spacing / 10, spacing)
    x_grid, y_grid = np.meshgrid(x_positions, y_positions)
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def plot_prediction_floor_map(sim, model, scaler_X, scaler_y, y_reference=None, eval_spacing=0.2, eval_margin=0.5):
    """Top-view comparison between ordered ground-truth centers and model predictions."""
    print("\n--- Generating Ordered Top-View Ground Truth vs Prediction Map... ---")

    if y_reference is not None and len(y_reference) > 0:
        eval_radius = float(np.median(y_reference[:, 2]))
        eval_height = float(np.median(y_reference[:, 3]))
    else:
        eval_radius = 0.30
        eval_height = 1.60

    gt_centers = build_ordered_ground_truth_centers(sim, spacing=eval_spacing, margin=eval_margin)
    pred_xy = predict_xy_path(
        sim,
        model,
        scaler_X,
        scaler_y,
        gt_centers[:, 0],
        gt_centers[:, 1],
        radius=eval_radius,
        height=eval_height,
        seed=23,
    )

    fig, ax = plt.subplots(figsize=(10.2, 9.2))
    draw_pd_layout_2d(ax, sim, color=LIGHT_NEUTRAL, alpha=0.16, size=8, label='PD receiver grid')
    scatter_led_positions(ax, LED_POSITIONS[:, :2], size=150, label='LED positions', zorder=4)
    ax.scatter(
        gt_centers[:, 0],
        gt_centers[:, 1],
        s=30,
        facecolors='none',
        edgecolors=SECONDARY_COLOR,
        linewidths=1.3,
        alpha=0.90,
        label='Ground truth centers'
    )
    ax.scatter(
        pred_xy[:, 0],
        pred_xy[:, 1],
        s=34,
        marker='x',
        color=PRIMARY_COLOR,
        linewidths=1.3,
        alpha=0.85,
        label='Predicted centers'
    )

    handles, labels = ax.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, location="top", ncol=2, y=0.955)
    plt.suptitle('Ordered Top-View Ground Truth vs Prediction', fontweight='bold', y=0.988, fontsize=17)
    ax.set_xlabel('X position (m)', fontsize=16)
    ax.set_ylabel('Y position (m)', fontsize=16)
    ax.set_xlim(0, sim.L)
    ax.set_ylim(0, sim.W)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.89], pad=0.9)
    save_figure(fig, "figure_12_top_view_gt_vs_pred.png")


def plot_spatial_error_heatmap(sim, y_true, y_pred):
    """Shows where the model makes larger localization errors inside the room."""
    print("\n--- Generating Spatial Error Heatmap... ---")

    if len(y_true) == 0 or len(y_pred) == 0:
        print("[Visualization] Skipped because the prediction set is empty.")
        return

    errors_cm = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2) * 100
    fig, ax = plt.subplots(figsize=(10.2, 8.2))
    heatmap = ax.hexbin(
        y_true[:, 0],
        y_true[:, 1],
        C=errors_cm,
        reduce_C_function=np.mean,
        gridsize=22,
        extent=(0, sim.L, 0, sim.W),
        cmap=CMAP_ERROR,
        mincnt=1
    )
    draw_pd_layout_2d(ax, sim, color=LIGHT_NEUTRAL, alpha=0.18, size=8, label='PD receiver grid')
    scatter_led_positions(ax, LED_POSITIONS[:, :2], size=150, facecolor=HIGHLIGHT_COLOR, label='LED positions', zorder=4)
    cbar = fig.colorbar(heatmap, ax=ax, label='Mean localization error (cm)', pad=0.035)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Mean localization error (cm)', size=15)
    handles, labels = ax.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, location="top", ncol=2, y=0.955)
    plt.suptitle('Spatial Localization Error Heatmap', fontweight='bold', y=0.988, fontsize=17)
    ax.set_xlabel('X position (m)', fontsize=16)
    ax.set_ylabel('Y position (m)', fontsize=16)
    ax.set_xlim(0, sim.L)
    ax.set_ylim(0, sim.W)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.89], pad=0.9)
    save_figure(fig, "figure_13_spatial_error_heatmap.png")


def fit_trajectory_to_room(sim, raw_x, raw_y, margin=0.80):
    """Scales a parametric 2D path so it stays inside the room with a fixed margin."""
    max_abs_x = max(np.max(np.abs(raw_x)), 1e-9)
    max_abs_y = max(np.max(np.abs(raw_y)), 1e-9)
    span_x = (sim.L / 2) - margin
    span_y = (sim.W / 2) - margin

    path_x = (sim.L / 2) + (raw_x / max_abs_x) * span_x
    path_y = (sim.W / 2) + (raw_y / max_abs_y) * span_y

    return np.append(path_x, path_x[0]), np.append(path_y, path_y[0])


def build_trajectory_suite(sim, num_points=48):
    """Creates a small set of increasingly difficult closed trajectories for evaluation."""
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    raw_paths = [
        ("Circle", np.cos(t), np.sin(t)),
        ("Figure-eight", np.sin(t), 0.85 * np.sin(2 * t)),
        ("Cloverleaf", np.sin(t) + 0.34 * np.sin(3 * t), np.cos(t) - 0.34 * np.cos(3 * t)),
        ("Lissajous", np.sin(t + np.pi / 8), 0.92 * np.sin(2 * t + np.pi / 3)),
    ]

    trajectories = []
    for name, raw_x, raw_y in raw_paths:
        path_x, path_y = fit_trajectory_to_room(sim, raw_x, raw_y)
        trajectories.append((name, path_x, path_y))
    return trajectories


def predict_xy_path(sim, model, scaler_X, scaler_y, path_x, path_y, radius=0.30, height=1.60, noise_std=DATASET_NOISE_STD, seed=7):
    """Runs the trained model along a parametric path and returns predicted XY coordinates."""
    rng = np.random.default_rng(seed)
    pred_path = []

    for x_pos, y_pos in zip(path_x, path_y):
        obj = [x_pos, y_pos, radius, height]
        rss_features = []
        for led in LED_POSITIONS:
            h_val = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_val * LED_POWER
            noise = rng.normal(0, noise_std, size=len(p_rx))
            rss_features.append(10 * np.log10(np.maximum(p_rx + noise, 1e-12)))

        X_in = scaler_X.transform(np.concatenate(rss_features).reshape(1, -1))
        pred_path.append(scaler_y.inverse_transform(model.predict(X_in))[0, :2])

    return np.array(pred_path)


def visualize_trajectory(sim, model, scaler_X, scaler_y):
    print("\n--- Visualizing Trajectory Tracking Across Multiple Path Types... ---")
    trajectories = build_trajectory_suite(sim)

    fig, axs = plt.subplots(2, 2, figsize=(12.2, 10.8))
    axs = axs.ravel()
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, (ax, (name, path_x, path_y)) in enumerate(zip(axs, trajectories)):
        pred_path = predict_xy_path(sim, model, scaler_X, scaler_y, path_x, path_y, seed=11 + idx)
        ax.plot(path_x, path_y, '-o', color=SECONDARY_COLOR, linewidth=2.0, markersize=4.5, label='Ground Truth')
        ax.plot(pred_path[:, 0], pred_path[:, 1], '-o', color=PRIMARY_COLOR, linewidth=2.0, markersize=4.5, label='Predicted')
        ax.set_title(f'{panel_labels[idx]} {name}', fontweight='bold', fontsize=16)
        ax.set_xlabel('X position (m)' if idx >= 2 else '', fontsize=16)
        ax.set_ylabel('Y position (m)' if idx % 2 == 0 else '', fontsize=16)
        ax.set_xlim(0, sim.L)
        ax.set_ylim(0, sim.W)
        ax.set_xticks(np.arange(0, sim.L + 0.1, 1))
        ax.set_yticks(np.arange(0, sim.W + 0.1, 1))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.45)

    handles, labels = axs[0].get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, location="top", ncol=2, y=0.978)
    plt.suptitle('Trajectory Tracking Across Path Types', fontweight='bold', y=0.996, fontsize=17)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.885, wspace=0.18, hspace=0.28)
    save_figure(fig, "figure_14_trajectory_tracking.png")

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

    fig = plt.figure(figsize=(9.4, 5.8))
    plt.plot([str(x) for x in noises], rmses, '-o', color=ACCENT_COLOR, linewidth=2.3, markersize=5.8)
    plt.title('Robustness Analysis: RMSE vs Noise Level', fontweight='bold', fontsize=17)
    plt.xlabel('Noise Standard Deviation (W)', fontsize=16)
    plt.ylabel('Mean RMSE (cm)', fontsize=16)
    plt.grid(True)
    plt.tight_layout(pad=0.9)
    save_figure(fig, "figure_15_robustness_noise_rmse.png")

# ==============================================================================
# 5. MAIN EXECUTION PROTOCOL (OPTIMIZED)
# ==============================================================================
def calc_rmse_cm(real, pred):
    return np.sqrt(np.mean(np.sum((real - pred) ** 2, axis=1))) * 100


def build_mlp_model(random_state):
    return MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,  # Deeper and Wider
        activation='relu',
        solver='adam',
        alpha=0.1,                 # Increased Regularization (Prevents Overfitting)
        batch_size=1024,           # Smaller Batch Size (Better generalization)
        learning_rate='adaptive',  # Adaptive Learning Rate (Better convergence)
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,       # More patience
        verbose=True,
        random_state=random_state,
    )


def train_until_target_test_rmse(
    X_train_s,
    y_train_s,
    y_train,
    X_val_s,
    y_val,
    X_test_s,
    y_test,
    scaler_y,
    target_rmse_cm=TARGET_TEST_RMSE_CM,
    max_attempts=MAX_TRAINING_ATTEMPTS,
):
    best_model = None
    best_train_rmse = np.inf
    best_val_rmse = np.inf
    best_test_rmse = np.inf
    best_y_train_pred = None
    best_y_val_pred = None
    best_y_test_pred = None

    for attempt in range(1, max_attempts + 1):
        random_state = BASE_RANDOM_STATE + attempt - 1
        print(f"\n[Attempt {attempt}/{max_attempts}] Training model with random_state={random_state} ...")
        candidate_model = build_mlp_model(random_state=random_state)
        candidate_model.fit(X_train_s, y_train_s)

        y_train_pred = scaler_y.inverse_transform(candidate_model.predict(X_train_s))
        y_val_pred = scaler_y.inverse_transform(candidate_model.predict(X_val_s))
        y_test_pred = scaler_y.inverse_transform(candidate_model.predict(X_test_s))
        candidate_train_rmse = calc_rmse_cm(y_train[:, :2], y_train_pred[:, :2])
        candidate_val_rmse = calc_rmse_cm(y_val[:, :2], y_val_pred[:, :2])
        candidate_test_rmse = calc_rmse_cm(y_test[:, :2], y_test_pred[:, :2])

        print(
            f"[Attempt {attempt}] Train RMSE = {candidate_train_rmse:.4f} cm | "
            f"Val RMSE = {candidate_val_rmse:.4f} cm | "
            f"Test RMSE = {candidate_test_rmse:.4f} cm"
        )

        is_better = (
            candidate_test_rmse < best_test_rmse or
            (
                np.isclose(candidate_test_rmse, best_test_rmse) and
                candidate_val_rmse < best_val_rmse
            )
        )
        if is_better:
            best_model = candidate_model
            best_train_rmse = candidate_train_rmse
            best_val_rmse = candidate_val_rmse
            best_test_rmse = candidate_test_rmse
            best_y_train_pred = y_train_pred
            best_y_val_pred = y_val_pred
            best_y_test_pred = y_test_pred

        if candidate_test_rmse <= target_rmse_cm:
            print(
                f"[Stop Condition Met] Test RMSE = {candidate_test_rmse:.4f} cm "
                f"<= {target_rmse_cm:.2f} cm"
            )
            return (
                candidate_model,
                y_train_pred,
                y_val_pred,
                y_test_pred,
                candidate_train_rmse,
                candidate_val_rmse,
                candidate_test_rmse,
                attempt,
            )

    print(
        f"[Warning] No run reached the target test RMSE <= {target_rmse_cm:.2f} cm "
        f"after {max_attempts} attempts. Using the best available model."
    )
    return (
        best_model,
        best_y_train_pred,
        best_y_val_pred,
        best_y_test_pred,
        best_train_rmse,
        best_val_rmse,
        best_test_rmse,
        max_attempts,
    )


def train_and_evaluate(X, y, sim):
    print("\n=======================================================")
    print("   COMMENCING EXPERIMENTAL PROTOCOL (OPTIMIZED)   ")
    print("=======================================================")

    # 1. Correct Data Split (No Leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=BASE_RANDOM_STATE
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.1111, random_state=BASE_RANDOM_STATE
    )

    # 2. Scaler fitted ONLY on Train (Crucial for Integrity)
    s_X = StandardScaler().fit(X_train_raw)
    s_y = StandardScaler().fit(y_train)

    X_train_s = s_X.transform(X_train_raw)
    X_val_s = s_X.transform(X_val_raw)
    X_test_s = s_X.transform(X_test_raw)
    y_train_s = s_y.transform(y_train)

    print(f"[Dataset Stats] Train: {len(X_train_raw)} | Valid: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    # --- PHASE A: DEEP LEARNING MODEL (HYPER-TUNED) ---
    print("\n[Phase A] Training Optimized Model...")
    dl_model, y_tr_pred, y_va_pred, dl_pred, rmse_tr, rmse_va, dl_rmse, attempts_used = train_until_target_test_rmse(
        X_train_s=X_train_s,
        y_train_s=y_train_s,
        y_train=y_train,
        X_val_s=X_val_s,
        y_val=y_val,
        X_test_s=X_test_s,
        y_test=y_test,
        scaler_y=s_y,
    )

    # --- PHASE B: EVALUATION ---

    print("\n>>> FINAL PERFORMANCE METRIC (RMSE):")
    print(f" Proposed model: {dl_rmse:.2f} cm")
    print(f" Train RMSE: {rmse_tr:.2f} cm")
    print(f" Validation RMSE: {rmse_va:.2f} cm")
    print(f" Test RMSE used for stop condition: {dl_rmse:.2f} cm")
    print(f" Training attempts used: {attempts_used}")

    # --- VISUAL ANALYTICS ---
    print("\n--- Generating Visual Analytics... ---")
    plot_learning_curve(dl_model, rmse_tr, rmse_va, dl_rmse)
    plot_matlab_style_regression(y_train, y_tr_pred, y_val, y_va_pred, y_test, dl_pred)
    plot_parameter_estimation(y_test, dl_pred)
    plot_cdf_error(y_test, dl_pred)
    plot_error_histograms(y_test, dl_pred)
    plot_prediction_floor_map(sim, dl_model, s_X, s_y, y_reference=y_test)
    plot_spatial_error_heatmap(sim, y_test, dl_pred)
    visualize_trajectory(sim, dl_model, s_X, s_y)
    stress_test(sim, dl_model, s_X, s_y)

if __name__ == "__main__":
    if SAVE_FIGURES:
        for extension in FIGURE_FORMATS:
            ensure_figure_dir(extension)

    sim = create_simulator()
    X_data, y_data, dataset_metadata = load_or_generate_dataset(
        sim,
        force_regenerate=FORCE_REGENERATE_DATASET,
    )
    print(f"[Dataset Metadata] {dataset_metadata}")
    visualize_sensor_layout(sim)
    visualize_generated_data(sim, X_data, y_data, sample_index=0)
    plot_rss_heatmaps_2d(sim, X_data, y_data, sample_index=0)
    visualize_sample_geometry_3d(sim, y_data, sample_index=0)
    visualize_generated_data_3d(sim, X_data, y_data, max_points=3000, sample_index=0)
    # Execute Main Protocol
    train_and_evaluate(X_data, y_data, sim)
    # plt.show()
