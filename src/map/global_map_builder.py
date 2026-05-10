import json
import numpy as np
import cv2

from pathlib import Path
from src.config import (
    DATASET_ROOT,
    GLOBAL_MAP_RESOLUTION,
    ROAD_LABEL,
    LANE_LABEL,
)


def world_to_pixel(x, y, x_min, y_min, resolution):
    u = int((x - x_min) / resolution)
    v = int((y - y_min) / resolution)
    return u, v


def build_global_semantic_map(world, map_name):
    carla_map = world.get_map()

    waypoints = carla_map.generate_waypoints(1.0)

    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]

    margin = 50.0

    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin

    width = int((x_max - x_min) / GLOBAL_MAP_RESOLUTION)
    height = int((y_max - y_min) / GLOBAL_MAP_RESOLUTION)

    global_map = np.zeros((height, width), dtype=np.uint8)

    for wp in waypoints:
        loc = wp.transform.location

        u, v = world_to_pixel(
            loc.x,
            loc.y,
            x_min,
            y_min,
            GLOBAL_MAP_RESOLUTION,
        )

        lane_width_px = max(1, int(wp.lane_width / GLOBAL_MAP_RESOLUTION / 2))

        if 0 <= u < width and 0 <= v < height:
            cv2.circle(
                global_map,
                (u, v),
                lane_width_px,
                ROAD_LABEL,
                thickness=-1,
            )

            cv2.circle(
                global_map,
                (u, v),
                1,
                LANE_LABEL,
                thickness=-1,
            )

    map_dir = DATASET_ROOT / "global_maps" / map_name
    map_dir.mkdir(parents=True, exist_ok=True)

    np.save(map_dir / "semantic_map.npy", global_map)

    preview = (global_map * 80).astype(np.uint8)
    cv2.imwrite(str(map_dir / "semantic_map.png"), preview)

    meta = {
        "map_name": map_name,
        "resolution": GLOBAL_MAP_RESOLUTION,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": width,
        "height": height,
        "labels": {
            "background": 0,
            "road": ROAD_LABEL,
            "lane": LANE_LABEL,
        },
    }

    with open(map_dir / "semantic_map_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved global semantic map to {map_dir}")