import json
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT, FOV, FIXED_DELTA_SECONDS


def create_scene_dirs(scene_dir):
    dirs = {
        "rgb_front": scene_dir / "rgb" / "front",
        "depth_front": scene_dir / "depth" / "front",
        "depth_value_front": scene_dir / "depth_value" / "front",
        "semantic_front": scene_dir / "semantic" / "front",
        "lidar_top": scene_dir / "lidar" / "top",
        "pose": scene_dir / "pose",
        "gnss": scene_dir / "gnss",
        "imu": scene_dir / "imu",
        "calib": scene_dir / "calib",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def save_meta(scene_dir, scene_id, map_name, weather_name, max_frames):
    meta = {
        "scene_id": scene_id,
        "map": map_name,
        "weather": weather_name,
        "fps": int(1 / FIXED_DELTA_SECONDS),
        "fixed_delta_seconds": FIXED_DELTA_SECONDS,
        "num_frames": max_frames,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "fov": FOV,
        "sensors": [
            "rgb_front",
            "depth_front",
            "semantic_front",
            "lidar_top",
            "pose",
            "gnss",
            "imu",
        ],
    }

    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)