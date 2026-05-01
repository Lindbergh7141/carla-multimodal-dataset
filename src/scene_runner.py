import carla
from src.config import SCENE_CONFIGS
from src.collect_scene import collect_scene  # 你现在的函数


class SceneRunner:
    def __init__(self):
        self.client = None

    def setup_client(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(120.0)

    def run_all(self):
        self.setup_client()

        for cfg in SCENE_CONFIGS:
            print("=" * 60)
            print(f"Running scene: {cfg['scene_id']}")
            print("=" * 60)

            collect_scene(
                client=self.client,
                scene_id=cfg["scene_id"],
                map_name=cfg["map"],
                weather_name=cfg["weather"],
                max_frames=cfg["max_frames"],
                spawn_index=cfg["spawn_index"],
            )