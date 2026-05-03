# CARLA Multimodal Dataset

A modular pipeline for generating **cross-visual, multimodal datasets** in CARLA, designed for **robust multi-sensor localization and perception research**.

---

##Overview

This project provides a **scalable and configurable data collection system** built on CARLA.

It supports synchronized acquisition of multiple modalities under diverse environments, including:

* RGB camera
* Depth camera
* Semantic segmentation
* LiDAR point cloud
* GNSS (GPS)
* IMU
* Ground truth pose

The goal is to build a dataset for:

> **Cross-visual multimodal fusion localization under complex conditions** and also other tasks about robotic and autonomous driving

---

##Key Features

* ✅ **Synchronized multi-sensor data collection**
* ✅ **Config-driven scene generation**
* ✅ **Multiple weather conditions (rain, fog, sunset, etc.)**
* ✅ **Modular architecture (SensorManager, DatasetWriter, SceneRunner)**
* ✅ **Per-scene metadata and indexing**
* 🚧 OSM / OpenDRIVE map integration (planned)
* 🚧 Calibration export (planned)

---

## 📂 Project Structure

```
src/
├── config/              # Scene and sensor configuration
├── dataset/             # Dataset writer (meta, index, dirs)
├── sensors/             # SensorManager (all sensors + sync)
├── utils/               # Weather and helper functions
├── runner/              # SceneRunner (execution control)
└── collect_scene.py     # Core data collection logic
```

---

## 📊 Dataset Structure

Each scene is saved as:

```
dataset/
└── scene_xxx/
    ├── meta.json
    ├── index.json
    ├── rgb/front/
    ├── depth/front/
    ├── semantic/front/
    ├── lidar/top/
    ├── gnss/
    ├── imu/
    └── pose/
```

* `meta.json`: scene-level configuration
* `index.json`: per-frame file mapping

---

## ⚙️ Installation

Requirements:

* Python 3.8+
* CARLA (>= 0.9.x)

```bash
git clone <repo>
cd carla-multimodal-dataset
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Start CARLA

```bash
./CarlaUE4.sh
```

### 2. Configure scenes

Edit:

```python
src/config/config.py
```

Example:

```python
SCENE_CONFIGS = [
    {
        "scene_id": "town03_rain",
        "map": "Town03",
        "weather": "rain_day",
        "max_frames": 1000,
        "spawn_index": 0,
    }
]
```

---

### 3. Run data collection

```bash
python main.py
```

---

## 🌦️ Supported Weather

* Clear day
* Cloudy
* Rain (soft / hard)
* Sunset

(Extendable via `WeatherParameters`)

---

## 🧩 Modular Design

This project separates responsibilities:

* `SceneRunner` → controls execution
* `collect_scene` → handles scene lifecycle
* `SensorManager` → manages sensors & synchronization
* `dataset_writer` → handles data storage

This design allows easy extension for:

* new sensors
* new weather conditions
* multi-camera setups
* large-scale data generation

---

##  Future Work

* [ ] OSM → OpenDRIVE map integration
* [ ] Sensor calibration export
* [ ] Raw depth & semantic label storage
* [ ] Multi-camera surround system
* [ ] Dataset format alignment (nuScenes / Waymo)
* [ ] Multi-process data collection

---

## 📌 Research Goal

This dataset is designed for:

> **Robust multimodal localization across visual domains and environmental changes**

---

## 📜 License

MIT License (or your choice)

---

## 👤 Author

Your Name

---
