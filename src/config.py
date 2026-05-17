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

SCENE_CONFIGS = []

train_towns = [
    "Town03",
]

val_towns = [
    "Town03",
]

test_towns = [
    "Town05",
]

weathers = [
    "clear_day",
    "cloudy_day",
]

# -------------------
# TRAIN
# -------------------

for town in train_towns:
    for weather in weathers:
        for spawn in range(5):

            SCENE_CONFIGS.append({
                "scene_id":
                f"train_{town.lower()}_{weather}_{spawn:03d}",

                "split": "train",

                "map": town,

                "weather": weather,

                "max_frames": 500,

                "spawn_index": spawn,

                "seed": 1000 + spawn,
            })

# -------------------
# VAL
# -------------------

for town in val_towns:
    for weather in weathers:
        for spawn in range(5, 7):

            SCENE_CONFIGS.append({
                "scene_id":
                f"val_{town.lower()}_{weather}_{spawn:03d}",

                "split": "val",

                "map": town,

                "weather": weather,

                "max_frames": 500,

                "spawn_index": spawn,

                "seed": 2000 + spawn,
            })

# -------------------
# TEST
# -------------------

for town in test_towns:
    for weather in weathers:
        for spawn in range(3):

            SCENE_CONFIGS.append({
                "scene_id":
                f"test_{town.lower()}_{weather}_{spawn:03d}",

                "split": "test",

                "map": town,

                "weather": weather,

                "max_frames": 500,

                "spawn_index": spawn,

                "seed": 3000 + spawn,
            })

print(
    f"Total scenes: "
    f"{len(SCENE_CONFIGS)}"
)