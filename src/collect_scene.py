import carla
import json
import queue
import numpy as np
import random

from src.config import (
    DATASET_ROOT,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_FOV,
    FIXED_DELTA_SECONDS,
)
from src.utils.weather import get_weather
from src.dataset.dataset_writer import create_scene_dirs, save_meta
from src.sensors.sensor_manager import SensorManager
from src.dataset.calibration_writer import save_calibration
from src.map.global_map_builder import build_global_semantic_map

def carla_depth_to_array(depth_image):
    """
    Convert CARLA depth image to metric depth in meters.
    Output shape: H x W
    """
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (depth_image.height, depth_image.width, 4))

    # CARLA raw image format is BGRA
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)

    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0)

    depth_meters = 1000.0 * normalized

    return depth_meters

def carla_semantic_to_array(semantic_image):
    """
    Convert CARLA semantic segmentation image to class id map.
    Output shape: H x W
    Each value is a semantic class id.
    """
    array = np.frombuffer(semantic_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (semantic_image.height, semantic_image.width, 4))

    # CARLA semantic label is stored in the R channel
    semantic_ids = array[:, :, 2]

    return semantic_ids

def collect_scene(client, scene_id, split, map_name, weather_name, max_frames, spawn_index, seed):
    print("=" * 60)
    print(f"Collecting {scene_id}: {map_name}, {weather_name}")
    print("=" * 60)

    scene_dir = DATASET_ROOT / split / scene_id
    dirs = create_scene_dirs(scene_dir)
    save_meta(scene_dir, scene_id, map_name, weather_name, max_frames)

    world = client.load_world(map_name)
    random.seed(seed)
    np.random.seed(seed)
    traffic_manager = client.get_trafficmanager(8000)

    traffic_manager.set_random_device_seed(seed)
    build_global_semantic_map(world, map_name)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    world.set_weather(get_weather(weather_name))

    blueprint_library = world.get_blueprint_library()

    actors = []
    index = {}

    try:
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_index % len(spawn_points)]

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        sensor_manager = SensorManager(world, blueprint_library, vehicle)
        sensor_manager.setup_sensors()

        save_calibration(
            calib_dir=dirs["calib"],
            sensor_specs=sensor_manager.sensor_specs,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            fov=IMAGE_FOV,
        )

        positions = []
        trajectory = []

        frame_count = 0

        while frame_count < max_frames:
            world.tick()

            frame_data = sensor_manager.get_frame()

            if frame_data is None:
                 continue

            image = frame_data["rgb_front"]
            depth_image = frame_data["depth_front"]
            semantic_image = frame_data["semantic_front"]
            semantic_bev_image = frame_data["semantic_bev"]
            points = frame_data["lidar_points"]
            gnss_data = frame_data["gnss"]
            imu_data = frame_data["imu"]


            frame_id = frame_count + 1
            frame_key = f"{frame_id:06d}"

            rgb_path = dirs["rgb_front"] / f"{frame_key}.png"
            image.save_to_disk(str(rgb_path))

            depth_path = dirs["depth_front"] / f"{frame_key}.png"
            depth_image.save_to_disk(
                str(depth_path),
                carla.ColorConverter.LogarithmicDepth,
            )
            depth_array = carla_depth_to_array(depth_image)
            depth_value_path = dirs["depth_value_front"] / f"{frame_key}.npy"
            np.save(str(depth_value_path), depth_array)


            semantic_path = dirs["semantic_front"] / f"{frame_key}.png"
            semantic_image.save_to_disk(
                str(semantic_path),
                carla.ColorConverter.CityScapesPalette,
            )
            semantic_array = carla_semantic_to_array(semantic_image)
            semantic_raw_path = dirs["semantic_raw_front"] / f"{frame_key}.npy"
            np.save(str(semantic_raw_path), semantic_array)

            semantic_bev_path = dirs["semantic_bev"] / f"{frame_key}.png"
            semantic_bev_image.save_to_disk(
                str(semantic_bev_path),
                carla.ColorConverter.CityScapesPalette,
            )
            bev_raw = carla_semantic_to_array(semantic_bev_image)
            bev_raw_path = dirs["bev_raw"] / f"{frame_key}.npy"
            np.save(str(bev_raw_path), bev_raw)

            road_mask = (bev_raw == 7).astype(np.uint8)
            lane_mask = (bev_raw == 6).astype(np.uint8)
            sidewalk_mask = (bev_raw == 8).astype(np.uint8)
            road_path = dirs["bev_road"] / f"{frame_key}.npy"
            lane_path = dirs["bev_lane"] / f"{frame_key}.npy"

            np.save(str(road_path), road_mask)
            np.save(str(lane_path), lane_mask)

            bev_multichannel = np.stack(
                [road_mask, lane_mask, sidewalk_mask],
                axis=0
            )
            bev_multichannel_path = dirs["bev_multichannel"] / f"{frame_key}.npy"
            np.save(str(bev_multichannel_path), bev_multichannel)

            lidar_path = dirs["lidar_top"] / f"{frame_key}.npy"
            np.save(str(lidar_path), points)

            transform = vehicle.get_transform()
            positions.append([
                transform.location.x,
                transform.location.y,
            ])
            trajectory.append({
                "frame": frame_id,
                "x": transform.location.x,
                "y": transform.location.y,
                "yaw": transform.rotation.yaw,
            })

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

            pose_path = dirs["pose"] / f"{frame_key}.json"
            with open(pose_path, "w") as f:
                json.dump(pose, f, indent=2)

            gnss_json = {
                "carla_frame": gnss_data.frame,
                "latitude": gnss_data.latitude,
                "longitude": gnss_data.longitude,
                "altitude": gnss_data.altitude,
            }

            gnss_path = dirs["gnss"] / f"{frame_key}.json"
            with open(gnss_path, "w") as f:
                json.dump(gnss_json, f, indent=2)

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

            imu_path = dirs["imu"] / f"{frame_key}.json"
            with open(imu_path, "w") as f:
                json.dump(imu_json, f, indent=2)

            index[frame_key] = {
                "rgb_front": str(rgb_path.relative_to(scene_dir)),
                "depth_front": str(depth_path.relative_to(scene_dir)),
                "depth_value_front": str(depth_value_path.relative_to(scene_dir)),
                "semantic_front": str(semantic_path.relative_to(scene_dir)),
                "semantic_raw_front": str(semantic_raw_path.relative_to(scene_dir)),
                "semantic_bev": str(semantic_bev_path.relative_to(scene_dir)),
                "bev_raw": str(bev_raw_path.relative_to(scene_dir)),
                "bev_multichannel": str(bev_multichannel_path.relative_to(scene_dir)),
                "lidar_top": str(lidar_path.relative_to(scene_dir)),
                "pose": str(pose_path.relative_to(scene_dir)),
                "gnss": str(gnss_path.relative_to(scene_dir)),
                "imu": str(imu_path.relative_to(scene_dir)),
                "carla_frame": image.frame,
            }
            
            print(f"{scene_id} saved frame {frame_key}")
            frame_count += 1

        total_distance = 0.0

        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]

            dx = x2 - x1
            dy = y2 - y1

            total_distance += (dx**2 + dy**2) ** 0.5

        print(f"Trajectory distance: {total_distance:.2f} meters")

        if total_distance < 20.0:
            import shutil

            shutil.rmtree(scene_dir)

            print("Discarded bad trajectory.")

            return
        
        trajectory_path = scene_dir / "trajectory.json"

        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        with open(scene_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    finally:
        print(f"Cleaning up {scene_id}...")

        if "sensor_manager" in locals():
            sensor_manager.destroy()

        for actor in actors:
            actor.destroy()

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print(f"{scene_id} done.")