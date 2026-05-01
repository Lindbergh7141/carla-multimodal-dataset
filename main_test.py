import carla
import time
import json
import queue
import numpy as np
from pathlib import Path


# =========================
# 基本配置
# =========================

DATASET_ROOT = Path("dataset")
SCENE_ID = "scene_000001"
SCENE_DIR = DATASET_ROOT / SCENE_ID

MAP_NAME = "Town03"
WEATHER_NAME = "clear_day"

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 90

MAX_FRAMES = 1000
FIXED_DELTA_SECONDS = 0.05  # 20 FPS


# =========================
# 数据集目录
# =========================

DIRS = {
    "rgb_front": SCENE_DIR / "rgb" / "front",
    "depth_front": SCENE_DIR / "depth" / "front",
    "semantic_front": SCENE_DIR / "semantic" / "front",
    "lidar_top": SCENE_DIR / "lidar" / "top",
    "pose": SCENE_DIR / "pose",
    "gnss": SCENE_DIR / "gnss",
    "imu": SCENE_DIR / "imu",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# =========================
# 连接 CARLA
# =========================

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)

world = client.load_world(MAP_NAME)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
world.apply_settings(settings)

blueprint_library = world.get_blueprint_library()


# =========================
# 天气
# =========================

WEATHERS = {
    "clear_day": carla.WeatherParameters.ClearNoon,
    "cloudy_day": carla.WeatherParameters.CloudyNoon,
    "wet_day": carla.WeatherParameters.WetNoon,
    "rain_day": carla.WeatherParameters.HardRainNoon,
    "clear_sunset": carla.WeatherParameters.ClearSunset,
}

world.set_weather(WEATHERS[WEATHER_NAME])


# =========================
# meta.json
# =========================

meta = {
    "scene_id": SCENE_ID,
    "map": MAP_NAME,
    "weather": WEATHER_NAME,
    "fps": int(1 / FIXED_DELTA_SECONDS),
    "fixed_delta_seconds": FIXED_DELTA_SECONDS,
    "num_frames": MAX_FRAMES,
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

with open(SCENE_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)


# =========================
# 创建 Actor
# =========================

actors = []
index = {}

try:
    # vehicle
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actors.append(vehicle)

    vehicle.set_autopilot(True)

    # RGB camera
    camera_transform = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=2.4),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actors.append(camera)

    # Depth camera
    depth_bp = blueprint_library.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    depth_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    depth_bp.set_attribute("fov", str(FOV))

    depth = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
    actors.append(depth)

    # Semantic camera
    semantic_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    semantic_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    semantic_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    semantic_bp.set_attribute("fov", str(FOV))

    semantic = world.spawn_actor(semantic_bp, camera_transform, attach_to=vehicle)
    actors.append(semantic)

    # LiDAR
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "50")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "56000")

    lidar_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=2.5),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    actors.append(lidar)

    # GNSS
    gnss_bp = blueprint_library.find("sensor.other.gnss")
    gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))

    gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
    actors.append(gnss)

    # IMU
    imu_bp = blueprint_library.find("sensor.other.imu")
    imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))

    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    actors.append(imu)

    # queues
    image_queue = queue.Queue()
    depth_queue = queue.Queue()
    semantic_queue = queue.Queue()
    lidar_queue = queue.Queue()
    gnss_queue = queue.Queue()
    imu_queue = queue.Queue()

    camera.listen(image_queue.put)
    depth.listen(depth_queue.put)
    semantic.listen(semantic_queue.put)
    lidar.listen(lidar_queue.put)
    gnss.listen(gnss_queue.put)
    imu.listen(imu_queue.put)

    frame_count = 0

    while frame_count < MAX_FRAMES:
        world.tick()

        image = image_queue.get()
        depth_image = depth_queue.get()
        semantic_image = semantic_queue.get()
        lidar_data = lidar_queue.get()
        gnss_data = gnss_queue.get()
        imu_data = imu_queue.get()

        if not (
            image.frame
            == depth_image.frame
            == semantic_image.frame
            == lidar_data.frame
            == gnss_data.frame
            == imu_data.frame
        ):
            print(
                "Frame mismatch:",
                image.frame,
                depth_image.frame,
                semantic_image.frame,
                lidar_data.frame,
                gnss_data.frame,
                imu_data.frame,
            )
            continue

        frame_id = frame_count + 1
        frame_key = f"{frame_id:06d}"

        # RGB
        rgb_path = DIRS["rgb_front"] / f"{frame_key}.png"
        image.save_to_disk(str(rgb_path))

        # Depth
        depth_path = DIRS["depth_front"] / f"{frame_key}.png"
        depth_image.save_to_disk(
            str(depth_path),
            carla.ColorConverter.LogarithmicDepth,
        )

        # Semantic
        semantic_path = DIRS["semantic_front"] / f"{frame_key}.png"
        semantic_image.save_to_disk(
            str(semantic_path),
            carla.ColorConverter.CityScapesPalette,
        )

        # LiDAR
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))

        lidar_path = DIRS["lidar_top"] / f"{frame_key}.npy"
        np.save(str(lidar_path), points)

        # Pose
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
            },
        }

        pose_path = DIRS["pose"] / f"{frame_key}.json"
        with open(pose_path, "w") as f:
            json.dump(pose, f, indent=2)

        # GNSS
        gnss_json = {
            "carla_frame": gnss_data.frame,
            "latitude": gnss_data.latitude,
            "longitude": gnss_data.longitude,
            "altitude": gnss_data.altitude,
        }

        gnss_path = DIRS["gnss"] / f"{frame_key}.json"
        with open(gnss_path, "w") as f:
            json.dump(gnss_json, f, indent=2)

        # IMU
        imu_json = {
            "carla_frame": imu_data.frame,
            "accelerometer": {
                "x": imu_data.accelerometer.x,
                "y": imu_data.accelerometer.y,
                "z": imu_data.accelerometer.z,
            },
            "gyroscope": {
                "x": imu_data.gyroscope.x,
                "y": imu_data.gyroscope.y,
                "z": imu_data.gyroscope.z,
            },
            "compass": imu_data.compass,
        }

        imu_path = DIRS["imu"] / f"{frame_key}.json"
        with open(imu_path, "w") as f:
            json.dump(imu_json, f, indent=2)

        # index
        index[frame_key] = {
            "rgb_front": str(rgb_path.relative_to(SCENE_DIR)),
            "depth_front": str(depth_path.relative_to(SCENE_DIR)),
            "semantic_front": str(semantic_path.relative_to(SCENE_DIR)),
            "lidar_top": str(lidar_path.relative_to(SCENE_DIR)),
            "pose": str(pose_path.relative_to(SCENE_DIR)),
            "gnss": str(gnss_path.relative_to(SCENE_DIR)),
            "imu": str(imu_path.relative_to(SCENE_DIR)),
            "carla_frame": image.frame,
        }

        print(f"saved frame {frame_key}")
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