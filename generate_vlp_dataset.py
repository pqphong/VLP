import argparse
import json
import os
import time

import numpy as np


DEFAULT_ROOM_DIM = (5.0, 5.0, 3.0)
DEFAULT_GRID_SIZE = 0.2
DEFAULT_N_SAMPLES = 50000
DEFAULT_MIN_VISIBLE_SENSORS = 1
DEFAULT_LED_POWER = 10.0
DEFAULT_SEMI_ANGLE = 60.0
DEFAULT_PD_AREA = 1e-4
DEFAULT_FOV = 60.0
DEFAULT_REFRACTIVE_INDEX = 1.5
DEFAULT_FILTER_GAIN = 1.0
DEFAULT_NOISE_STD = 1e-8
DEFAULT_CACHE_DIR = "cached_datasets"
DEFAULT_LED_POSITIONS = np.array([
    [1.25, 1.25, 3.0],
    [3.75, 1.25, 3.0],
    [3.75, 3.75, 3.0],
    [1.25, 3.75, 3.0],
], dtype=float)


def compute_lambertian_order(semi_angle_deg):
    return -np.log(2.0) / np.log(np.cos(np.radians(semi_angle_deg)))


def compute_concentrator_gain(refractive_index, fov_deg):
    return (refractive_index ** 2) / (np.sin(np.radians(fov_deg)) ** 2)


def _tokenize_value(value):
    text = str(value)
    for old, new in ((".", "p"), ("-", "m"), (" ", ""), (",", "_"), ("[", ""), ("]", ""), ("(", ""), (")", "")):
        text = text.replace(old, new)
    return text


def build_default_dataset_path(
    base_dir=DEFAULT_CACHE_DIR,
    room_dim=DEFAULT_ROOM_DIM,
    grid_size=DEFAULT_GRID_SIZE,
    n_samples=DEFAULT_N_SAMPLES,
    min_visible_sensors=DEFAULT_MIN_VISIBLE_SENSORS,
    led_power=DEFAULT_LED_POWER,
    semi_angle_deg=DEFAULT_SEMI_ANGLE,
):
    room_token = "x".join(_tokenize_value(v) for v in room_dim)
    filename = (
        f"vlp_room_{room_token}"
        f"_grid_{_tokenize_value(grid_size)}"
        f"_samples_{int(n_samples)}"
        f"_minvis_{int(min_visible_sensors)}"
        f"_power_{_tokenize_value(led_power)}"
        f"_semi_{_tokenize_value(semi_angle_deg)}.npz"
    )
    return os.path.join(base_dir, filename)


class VLPSimulator:
    def __init__(
        self,
        room_dim,
        grid_size,
        fov_deg,
        pd_area,
        lambertian_order,
        filter_gain,
        concentrator_gain,
    ):
        self.L, self.W, self.H = room_dim
        self.grid_size = grid_size
        self.fov_deg = fov_deg
        self.pd_area = pd_area
        self.lambertian_order = lambertian_order
        self.filter_gain = filter_gain
        self.concentrator_gain = concentrator_gain

        x = np.arange(0, self.L + grid_size / 100, grid_size)
        y = np.arange(0, self.W + grid_size / 100, grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.rx_coords = np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel(), np.zeros(self.X_grid.size)))
        self.n_sensors = self.rx_coords.shape[0]
        print(f"[System Init] Sensor Grid Initialized: {self.n_sensors} nodes (Resolution: {grid_size}m)")

    def calculate_los_channel(self, led_pos, rx_pos_list, obj_params=None):
        vec_d = rx_pos_list - led_pos
        dist = np.linalg.norm(vec_d, axis=1)
        dist_sq = dist ** 2

        cos_phi = -vec_d[:, 2] / dist
        cos_psi = -vec_d[:, 2] / dist

        h = np.zeros(len(dist))

        fov_rad = np.radians(self.fov_deg)
        valid_indices = (cos_psi >= np.cos(fov_rad)) & (cos_phi > 0)

        const_factor = ((self.lambertian_order + 1) * self.pd_area) / (2 * np.pi)
        h[valid_indices] = (
            (const_factor / dist_sq[valid_indices])
            * (cos_phi[valid_indices] ** self.lambertian_order)
            * cos_psi[valid_indices]
            * self.filter_gain
            * self.concentrator_gain
        )

        if obj_params is not None:
            ox, oy, r, h_obj = obj_params
            p1 = led_pos[:2]
            p2 = rx_pos_list[:, :2]
            d_vec = p2 - p1
            f_vec = p1 - np.array([ox, oy])

            a = np.sum(d_vec ** 2, axis=1)
            b = np.sum(d_vec * f_vec, axis=1)
            c = np.sum(f_vec ** 2) - r ** 2
            delta = b ** 2 - a * c
            potential_idx = np.where(delta >= 0)

            if len(potential_idx[0]) > 0:
                sqrt_delta = np.sqrt(delta[potential_idx])
                t1 = (-b[potential_idx] - sqrt_delta) / a[potential_idx]
                t2 = (-b[potential_idx] + sqrt_delta) / a[potential_idx]

                t_start = np.maximum(0, t1)
                t_end = np.minimum(1, t2)
                valid_intersections = t_start < t_end

                real_idx = potential_idx[0][valid_intersections]
                t_end_real = t_end[valid_intersections]

                z_at_exit = self.H * (1 - t_end_real)
                is_blocked = z_at_exit < h_obj

                h[real_idx[is_blocked]] = 0.0

        return h


def build_dataset_metadata(
    room_dim,
    grid_size,
    n_samples,
    min_visible_sensors,
    led_positions,
    led_power,
    semi_angle_deg,
    noise_std,
):
    n_x = len(np.arange(0, room_dim[0] + grid_size / 100, grid_size))
    n_y = len(np.arange(0, room_dim[1] + grid_size / 100, grid_size))
    return {
        "room_dim": [float(v) for v in room_dim],
        "grid_size": float(grid_size),
        "n_sensors": int(n_x * n_y),
        "n_samples": int(n_samples),
        "min_visible_sensors": int(min_visible_sensors),
        "led_positions": np.asarray(led_positions, dtype=float).tolist(),
        "led_power": float(led_power),
        "semi_angle_deg": float(semi_angle_deg),
        "noise_std": float(noise_std),
    }


def save_dataset(path, X, y, metadata):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        metadata=json.dumps(metadata),
    )
    print(f"[Dataset Saved] {path}")


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        metadata = json.loads(str(data["metadata"]))
    print(f"[Dataset Loaded] {path}")
    print(f"[Dataset Shape] X={X.shape}, y={y.shape}")
    return X, y, metadata


def generate_training_data(
    sim,
    n_samples,
    led_positions,
    led_power,
    semi_angle_deg,
    min_visible_sensors=DEFAULT_MIN_VISIBLE_SENSORS,
    noise_std=DEFAULT_NOISE_STD,
):
    X = []
    y = []
    print(f"--- Initiating Data Generation Protocol ({n_samples} samples) ---")
    start_time = time.time()
    count = 0
    cos_threshold = np.cos(np.radians(semi_angle_deg))

    while count < n_samples:
        ox = np.random.uniform(0.5, sim.L - 0.5)
        oy = np.random.uniform(0.5, sim.W - 0.5)
        r = np.random.uniform(0.15, 0.40)
        h = np.random.uniform(1.2, 1.9)
        obj = [ox, oy, r, h]

        visible_led_count = 0
        obj_top = np.array([ox, oy, h])
        for led in led_positions:
            vec = led - obj_top
            if abs(vec[2]) / np.linalg.norm(vec) > cos_threshold:
                visible_led_count += 1

        if visible_led_count < min_visible_sensors:
            continue

        rss_features = []
        for led in led_positions:
            h_channel = sim.calculate_los_channel(led, sim.rx_coords, obj_params=obj)
            p_rx = h_channel * led_power
            noise = np.random.normal(0, noise_std, size=len(p_rx))
            p_rx_noisy = np.maximum(p_rx + noise, 1e-12)
            rss_features.append(10 * np.log10(p_rx_noisy))

        X.append(np.concatenate(rss_features))
        y.append(obj)
        count += 1
        if count % 5000 == 0:
            print(f"   > Generated {count}/{n_samples} samples...")

    elapsed = time.time() - start_time
    print(f"[Data Gen] Completed in {elapsed:.2f}s")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def get_or_create_dataset(
    path,
    sim,
    n_samples,
    led_positions,
    led_power,
    semi_angle_deg,
    min_visible_sensors=DEFAULT_MIN_VISIBLE_SENSORS,
    noise_std=DEFAULT_NOISE_STD,
    force_regenerate=False,
):
    if os.path.exists(path) and not force_regenerate:
        X, y, metadata = load_dataset(path)
        expected_feature_count = sim.n_sensors * len(led_positions)
        cached_feature_count = int(X.shape[1]) if X.ndim == 2 else -1

        if cached_feature_count == expected_feature_count:
            return X, y, metadata

        print(
            "[Dataset Cache] Cache mismatch detected. "
            f"Expected features={expected_feature_count}; "
            f"found features={cached_feature_count}. Regenerating."
        )

    metadata = build_dataset_metadata(
        room_dim=(sim.L, sim.W, sim.H),
        grid_size=sim.grid_size,
        n_samples=n_samples,
        min_visible_sensors=min_visible_sensors,
        led_positions=led_positions,
        led_power=led_power,
        semi_angle_deg=semi_angle_deg,
        noise_std=noise_std,
    )
    X, y = generate_training_data(
        sim=sim,
        n_samples=n_samples,
        led_positions=led_positions,
        led_power=led_power,
        semi_angle_deg=semi_angle_deg,
        min_visible_sensors=min_visible_sensors,
        noise_std=noise_std,
    )
    save_dataset(path, X, y, metadata)
    return X, y, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and cache a VLP training dataset.")
    parser.add_argument("--samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of generated samples.")
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE, help="Sensor grid resolution in meters.")
    parser.add_argument("--output", type=str, default="", help="Optional output .npz path.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the cache file already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    lambertian_order = compute_lambertian_order(DEFAULT_SEMI_ANGLE)
    concentrator_gain = compute_concentrator_gain(DEFAULT_REFRACTIVE_INDEX, DEFAULT_FOV)
    sim = VLPSimulator(
        room_dim=DEFAULT_ROOM_DIM,
        grid_size=args.grid_size,
        fov_deg=DEFAULT_FOV,
        pd_area=DEFAULT_PD_AREA,
        lambertian_order=lambertian_order,
        filter_gain=DEFAULT_FILTER_GAIN,
        concentrator_gain=concentrator_gain,
    )

    output_path = args.output or build_default_dataset_path(
        room_dim=DEFAULT_ROOM_DIM,
        grid_size=args.grid_size,
        n_samples=args.samples,
        min_visible_sensors=DEFAULT_MIN_VISIBLE_SENSORS,
        led_power=DEFAULT_LED_POWER,
        semi_angle_deg=DEFAULT_SEMI_ANGLE,
    )

    X, y, _ = get_or_create_dataset(
        path=output_path,
        sim=sim,
        n_samples=args.samples,
        led_positions=DEFAULT_LED_POSITIONS,
        led_power=DEFAULT_LED_POWER,
        semi_angle_deg=DEFAULT_SEMI_ANGLE,
        min_visible_sensors=DEFAULT_MIN_VISIBLE_SENSORS,
        noise_std=DEFAULT_NOISE_STD,
        force_regenerate=args.force,
    )
    print(f"[Done] Cached dataset ready at: {output_path}")
    print(f"[Done] X shape={X.shape}, y shape={y.shape}")


if __name__ == "__main__":
    main()
