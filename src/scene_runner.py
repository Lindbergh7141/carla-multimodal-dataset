from collections import defaultdict
import carla

from src.config import SCENE_CONFIGS
from src.collect_scene import collect_scene


class SceneRunner:
    def __init__(self):
        self.client = None

    def setup_client(self):
        self.client = carla.Client(
            "localhost",
            2000
        )
        self.client.set_timeout(120.0)

    def run_all(self):

        self.setup_client()

        # ------------------
        # group by town
        # ------------------

        grouped = defaultdict(list)

        for cfg in SCENE_CONFIGS:
            grouped[cfg["map"]].append(cfg)

        # ------------------
        # load one town once
        # ------------------

        for map_name, scenes in grouped.items():

            print("=" * 60)
            print(f"Loading map: {map_name}")
            print("=" * 60)

            self.client.load_world(map_name)

            for cfg in scenes:

                print("=" * 60)
                print(
                    f"Running scene: "
                    f"{cfg['scene_id']}"
                )
                print("=" * 60)

                collect_scene(
                    client=self.client,
                    scene_id=cfg["scene_id"],
                    split=cfg["split"],
                    map_name=cfg["map"],
                    weather_name=cfg["weather"],
                    max_frames=cfg["max_frames"],
                    spawn_index=cfg["spawn_index"],
                    seed=cfg["seed"]
                )