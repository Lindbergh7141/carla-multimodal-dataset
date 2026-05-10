from pathlib import Path

DATASET_ROOT = Path("dataset")

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
IMAGE_FOV = 90





BEV_FOV = 90
BEV_SIZE = 512
BEV_RESOLUTION = 0.2
BEV_EXTENT_METERS = BEV_SIZE * BEV_RESOLUTION
BEV_EGO_CENTERED = True
BEV_CAMERA_HEIGHT = 50.0

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