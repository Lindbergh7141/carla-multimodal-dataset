import queue
import carla
import numpy as np

from src.config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_FOV, BEV_FOV, BEV_SIZE

class SensorManager:
    def __init__(self, world, blueprint_library, vehicle):
        self.world = world
        self.blueprint_library = blueprint_library
        self.vehicle = vehicle
        self.sensors = []
        self.queues = {}
        self.sensor_specs = {}

    def setup_sensors(self):
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        bev_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=50.0),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )


        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.5),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        small_sensor_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.0)
        )

        self._create_camera(
            name="rgb_front",
            blueprint_id="sensor.camera.rgb",
            transform=camera_transform,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            fov=IMAGE_FOV,
        )

        self._create_camera(
            name="depth_front",
            blueprint_id="sensor.camera.depth",
            transform=camera_transform,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            fov=IMAGE_FOV,
        )

        self._create_camera(
            name="semantic_front",
            blueprint_id="sensor.camera.semantic_segmentation",
            transform=camera_transform,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            fov=IMAGE_FOV,
        )


        self._create_camera(
            name="semantic_bev",
            blueprint_id="sensor.camera.semantic_segmentation",
            transform=bev_transform,
            image_width=BEV_SIZE,
            mage_height=BEV_SIZE,
            fov=BEV_FOV,
        )

        self._create_lidar(
            name="lidar_top",
            transform=lidar_transform,
        )

        self._create_other_sensor(
            name="gnss",
            blueprint_id="sensor.other.gnss",
            transform=small_sensor_transform,
        )

        self._create_other_sensor(
            name="imu",
            blueprint_id="sensor.other.imu",
            transform=small_sensor_transform,
        )

    def _create_camera(
        self,
        name,
        blueprint_id,
        transform,
        image_width,
        image_height,
        fov，
    ):
        bp = self.blueprint_library.find(blueprint_id)
        bp.set_attribute("image_size_x", str(image_width))
        bp.set_attribute("image_size_y", str(image_height))
        bp.set_attribute("fov", str(FOV))

        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        q = queue.Queue()
        sensor.listen(q.put)

        self.sensors.append(sensor)
        self.queues[name] = q
        self.sensor_specs[name] = {
            "type": "camera",
            "blueprint_id": blueprint_id,
            "transform": transform,
        }

    def _create_lidar(self, name, transform):
        bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        bp.set_attribute("range", "50")
        bp.set_attribute("rotation_frequency", "20")
        bp.set_attribute("channels", "32")
        bp.set_attribute("points_per_second", "56000")

        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        q = queue.Queue()
        sensor.listen(q.put)

        self.sensors.append(sensor)
        self.queues[name] = q
        self.sensor_specs[name] = {
            "type": "camera",
            "blueprint_id": "sensor.lidar.ray_cast",
            "transform": transform,
        }

    def _create_other_sensor(self, name, blueprint_id, transform):
        bp = self.blueprint_library.find(blueprint_id)

        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        q = queue.Queue()
        sensor.listen(q.put)

        self.sensors.append(sensor)
        self.queues[name] = q

        self.sensor_specs[name] = {
            "type": "other",
            "blueprint_id": blueprint_id,
            "transform": transform,
        }

    def get_frame(self):
        data = {
            "rgb_front": self.queues["rgb_front"].get(),
            "depth_front": self.queues["depth_front"].get(),
            "semantic_front": self.queues["semantic_front"].get(),
            "semantic_bev": self.queues["semantic_bev"].get(),
            "lidar_top": self.queues["lidar_top"].get(),
            "gnss": self.queues["gnss"].get(),
            "imu": self.queues["imu"].get(),
        }

        frames = [v.frame for v in data.values()]

        if len(set(frames)) != 1:
            print("Frame mismatch:", frames)
            return None

        lidar_raw = data["lidar_top"]
        points = np.frombuffer(lidar_raw.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))

        data["lidar_points"] = points
        data["carla_frame"] = frames[0]

        return data

    def destroy(self):
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()