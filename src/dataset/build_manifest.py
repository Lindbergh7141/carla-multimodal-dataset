import json
from pathlib import Path

from src.config import DATASET_ROOT


def build_manifest():

    manifest = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_dir = (
            DATASET_ROOT / split
        )

        if not split_dir.exists():
            continue

        #  scene
        for scene_dir in split_dir.iterdir():

            if not scene_dir.is_dir():
                continue

            print(
                f"Processing "
                f"{scene_dir.name}"
            )

            # -------------------
            # meta
            # -------------------

            meta_path = (
                scene_dir
                / "meta.json"
            )

            with open(meta_path) as f:
                meta = json.load(f)

            map_name = meta["map"]

            # -------------------
            # calibration
            # -------------------

            calibration_path = (
                scene_dir
                / "calibration"
                / "calibration.json"
            )

            # -------------------
            # global map
            # -------------------

            global_map_path = (
                DATASET_ROOT
                / "global_maps"
                / map_name
                / "semantic_map.npy"
            )

            global_map_meta_path = (
                DATASET_ROOT
                / "global_maps"
                / map_name
                / "semantic_map_meta.json"
            )

            # -------------------
            # trajectory
            # -------------------

            trajectory_path = (
                scene_dir
                / "trajectory.json"
            )

            with open(
                trajectory_path
            ) as f:

                trajectory = json.load(f)

            for pose in trajectory:

                frame_id = (
                    pose["frame"]
                )

                frame_key = (
                    f"{frame_id:06d}"
                )

                rgb_path = (
                    scene_dir
                    / "rgb/front"
                    / f"{frame_key}.png"
                )

                bev_path = (
                    scene_dir
                    / "bev/multichannel"
                    / f"{frame_key}.npy"
                )

                # 跳过坏文件
                if (
                    not rgb_path.exists()
                ):
                    continue

                sample = {

                    "sample_id":
                    f"{scene_dir.name}"
                    f"_{frame_key}",

                    "split":
                    split,

                    "scene_id":
                    scene_dir.name,

                    "map":
                    map_name,

                    "image":
                    str(rgb_path),

                    "semantic_bev":
                    str(bev_path),

                    "pose":
                    {
                        "x":
                        pose["x"],

                        "y":
                        pose["y"],

                        "yaw":
                        pose["yaw"],
                    },

                    "global_map":
                    str(
                        global_map_path
                    ),

                    "global_map_meta":
                    str(global_map_meta_path),

                    "calibration":
                    str(
                        calibration_path
                    ),
                }

                manifest.append(
                    sample
                )

    save_path = (
        DATASET_ROOT
        / "manifest.json"
    )

    with open(
        save_path,
        "w"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print(
        f"Saved "
        f"{len(manifest)} "
        f"samples"
    )


if __name__ == "__main__":
    build_manifest()