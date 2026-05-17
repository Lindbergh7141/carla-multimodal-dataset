import os
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


SAVE_DIR = (
    "/cvhci/temp/yhuang/"
    "datasets/sanity_check"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

MANIFEST_PATH = (
    "/cvhci/temp/yhuang/"
    "datasets/manifest.json"
)


def world_to_pixel(
    x,
    y,
    x_min,
    y_min,
    resolution
):
    u = int(
        (x - x_min)
        / resolution
    )

    v = int(
        (y - y_min)
        / resolution
    )

    return u, v


def main():

    with open(
        MANIFEST_PATH
    ) as f:

        manifest = json.load(f)

    samples = random.sample(
        manifest,
        20
    )

    for i, sample in enumerate(samples):

        print(
            f"[{i+1}/20]",
            sample["sample_id"]
        )

        image = cv2.imread(
            sample["image"]
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        global_map = np.load(
            sample["global_map"]
        )

        with open(
            sample[
                "global_map_meta"
            ]
        ) as f:

            meta = json.load(f)

        x = sample["pose"]["x"]
        y = sample["pose"]["y"]
        yaw = sample["pose"]["yaw"]

        u, v = world_to_pixel(
            x,
            y,
            meta["x_min"],
            meta["y_min"],
            meta["resolution"]
        )

        patch_size = 200

        u0 = max(0, u - patch_size)
        u1 = min(
            global_map.shape[1],
            u + patch_size
        )

        v0 = max(0, v - patch_size)
        v1 = min(
            global_map.shape[0],
            v + patch_size
        )

        patch = global_map[
            v0:v1,
            u0:u1
        ]

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(12, 6)
        )

        ax[0].imshow(image)
        ax[0].set_title("RGB")

        ax[1].imshow(patch)

        center_u = u - u0
        center_v = v - v0

        ax[1].scatter(
            center_u,
            center_v,
            c="red",
            s=100
        )

        yaw_rad = np.deg2rad(yaw)

        dx = (
            50
            * np.cos(yaw_rad)
        )

        dy = (
            -50
            * np.sin(yaw_rad)
        )

        ax[1].arrow(
            center_u,
            center_v,
            dx,
            dy,
            color="yellow",
            linewidth=4,
            head_width=10,
            head_length=10
        )

        ax[1].set_title(
            "Map Alignment"
        )

        save_path = (
            f"{SAVE_DIR}/"
            f"{sample['sample_id']}.png"
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    print(
        f"\nSaved to:\n"
        f"{SAVE_DIR}"
    )


if __name__ == "__main__":
    main()