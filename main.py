import carla
import time
import os
import json
import queue
import numpy as np
from pathlib import Path

DATASET_ROOT = Path("dataset")
SCENE_ID = "scene_000001"
SCENE_DIR = DATASET_ROOT / SCENE_ID

DIRS = {
    "rgb_front": SCENE_DIR / "rgb" / "front",
    "depth_front": SCENE_DIR / "depth" / "front",
    "semantic_front": SCENE_DIR / "semantic" / "front",
    "lidar_top": SCENE_DIR / "lidar" / "top",
    "pose": SCENE_DIR / "pose",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

#OUTPUT_DIR = "output"  测试模态采集
#os.makedirs(OUTPUT_DIR, exist_ok=True)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

meta = {
    "scene_id": SCENE_ID,
    "map": world.get_map().name,
    "fps": 20,
    "fixed_delta_seconds": 0.05,
    "sensors": [
        "rgb_front",
        "depth_front",
        "semantic_front",
        "lidar_top",
        "pose"
    ]
}

with open(SCENE_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# 开启同步模式
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

blueprint_library = world.get_blueprint_library()

actors = []

try:
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actors.append(vehicle)

    vehicle.set_autopilot(True)

    # RGB Camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.4)
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )
    actors.append(camera)

    # Depth Camera
    depth_bp = blueprint_library.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", "800")
    depth_bp.set_attribute("image_size_y", "600")
    depth_bp.set_attribute("fov", "90")

    depth = world.spawn_actor(
     depth_bp,
     camera_transform,
    attach_to=vehicle
    ) 
    actors.append(depth)

    # Semantic Segmentation Camera
    semantic_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    semantic_bp.set_attribute("image_size_x", "800")
    semantic_bp.set_attribute("image_size_y", "600")
    semantic_bp.set_attribute("fov", "90")

    semantic = world.spawn_actor(
      semantic_bp,
      camera_transform,
      attach_to=vehicle
    )
    actors.append(semantic)

    # LiDAR
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "50")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "56000")

    lidar_transform = carla.Transform(
        carla.Location(x=0.0, z=2.5)
    )

    lidar = world.spawn_actor(
        lidar_bp,
        lidar_transform,
        attach_to=vehicle
    )
    actors.append(lidar)

    image_queue = queue.Queue()
    depth_queue = queue.Queue()
    semantic_queue = queue.Queue()
    lidar_queue = queue.Queue()

    camera.listen(image_queue.put)
    depth.listen(depth_queue.put)
    semantic.listen(semantic_queue.put)
    lidar.listen(lidar_queue.put)

    frame_count = 0
    max_frames = 20

    index = {}

    while frame_count < max_frames:
        world.tick()

        image = image_queue.get()
        depth_image = depth_queue.get()
        semantic_image = semantic_queue.get()
        lidar_data = lidar_queue.get()

        # 保证同步
        if not (image.frame == depth_image.frame == semantic_image.frame == lidar_data.frame):
            print("Frame mismatch:", image.frame, depth_image.frame, semantic_image.frame, lidar_data.frame)
            continue

        frame_id = frame_count + 1

        # 保存 RGB
        rgb_path = DIRS["rgb_front"] / f"{frame_id:06d}.png"
        image.save_to_disk(str(rgb_path))

        depth_path = DIRS["depth_front"] / f"{frame_id:06d}.png"
        depth_image.save_to_disk(
            str(depth_path),
            carla.ColorConverter.LogarithmicDepth
        )

        semantic_path = DIRS["semantic_front"] / f"{frame_id:06d}.png"
        semantic_image.save_to_disk(
            str(semantic_path),
            carla.ColorConverter.CityScapesPalette
        )

        # 保存 LiDAR
        points = np.frombuffer(
            lidar_data.raw_data,
            dtype=np.float32
        )
        points = np.reshape(points, (-1, 4))
        lidar_path = DIRS["lidar_top"] / f"{frame_id:06d}.npy"
        np.save(str(lidar_path), points)

        # 保存 Pose
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        pose = {
            "carla_frame": image.frame,
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
            "velocity": {
                "x": velocity.x,
                "y": velocity.y,
                "z": velocity.z,
            }
        }

        pose_path = DIRS["pose"] / f"{frame_id:06d}.json"

        with open(pose_path, "w") as f:
            json.dump(pose, f, indent=2)

        frame_key = f"{frame_id:06d}"

        index[frame_key] = {
        "rgb_front": str(rgb_path.relative_to(SCENE_DIR)),
        "depth_front": str(depth_path.relative_to(SCENE_DIR)),
        "semantic_front": str(semantic_path.relative_to(SCENE_DIR)),
        "lidar_top": str(lidar_path.relative_to(SCENE_DIR)),
        "pose": str(pose_path.relative_to(SCENE_DIR)),
        "carla_frame": image.frame
        }

        print(f"saved frame {frame_id:06d}")

        frame_count += 1

    with open(SCENE_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

finally:
    print("Cleaning up...")

    for actor in actors:
        actor.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("Done.")