from pathlib import Path

DATASET_ROOT = Path("dataset")

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 90

FIXED_DELTA_SECONDS = 0.05  # 20 FPS

SCENE_CONFIGS = [
    {
        "scene_id": "scene_test",
        "map": "Town03",
        "weather": "clear_day",
        "max_frames": 20,
        "spawn_index": 0,
    }
]