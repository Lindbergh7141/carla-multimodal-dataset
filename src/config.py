from pathlib import Path

DATASET_ROOT = Path("/cvhci/temp/yhuang/datasets")

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

GLOBAL_MAP_RESOLUTION = 0.2
ROAD_LABEL = 1
LANE_LABEL = 2
SIDEWALK_LABEL = 3

SCENE_CONFIGS = [

    # =========================
    # TRAIN
    # =========================

    {
        "scene_id": "train_town03_clear_001",
        "split": "train",
        "map": "Town03",
        "weather": "clear_day",
        "max_frames": 300,
        "spawn_index": 0,
        "seed": 1001,
    },

    {
        "scene_id": "train_town03_clear_002",
        "split": "train",
        "map": "Town03",
        "weather": "clear_day",
        "max_frames": 300,
        "spawn_index": 10,
        "seed": 1002,
    },

    {
        "scene_id": "train_town03_cloudy_001",
        "split": "train",
        "map": "Town03",
        "weather": "cloudy_day",
        "max_frames": 300,
        "spawn_index": 20,
        "seed": 1003,
    },

    # =========================
    # VALIDATION
    # =========================

    {
        "scene_id": "val_town03_rain_001",
        "split": "val",
        "map": "Town03",
        "weather": "rain_day",
        "max_frames": 300,
        "spawn_index": 50,
        "seed": 2001,
    },

    # =========================
    # TEST
    # =========================

    {
        "scene_id": "test_town05_clear_001",
        "split": "test",
        "map": "Town05",
        "weather": "clear_day",
        "max_frames": 300,
        "spawn_index": 0,
        "seed": 3001,
    },

    {
        "scene_id": "test_town10_clear_001",
        "split": "test",
        "map": "Town10HD",
        "weather": "clear_day",
        "max_frames": 300,
        "spawn_index": 10,
        "seed": 3002,
    },
]