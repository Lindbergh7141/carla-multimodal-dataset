import json
import math
import numpy as np


def transform_to_dict(transform):
    return {
        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
        },
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
        },
        "matrix": np.array(transform.get_matrix()).tolist(),
        "inverse_matrix": np.array(transform.get_inverse_matrix()).tolist(),
    }


def camera_intrinsic(width, height, fov):
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))

    return {
        "width": width,
        "height": height,
        "fov": fov,
        "fx": f,
        "fy": f,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "K": [
            [f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
    }


def save_calibration(calib_dir, sensor_specs, image_width, image_height, fov):
    calib = {}

    for name, spec in sensor_specs.items():
        calib[name] = {
            "type": spec["type"],
            "blueprint_id": spec["blueprint_id"],
            "extrinsic": transform_to_dict(spec["transform"]),
        }

        if spec["type"] == "camera":
            calib[name]["intrinsic"] = camera_intrinsic(
                image_width,
                image_height,
                fov,
            )

    with open(calib_dir / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)