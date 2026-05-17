# Dataset Protocol v1.0

## Purpose

This dataset is designed for:

- Cross-view localization
- BEV-map matching
- Multi-modal perception
- Orientation estimation using OrienterNet

---

## Sensor Setup

### RGB Camera

- Name: rgb_front
- Resolution: 800 × 600
- FOV: 90°
- Position:
  - x = 1.5
  - y = 0.0
  - z = 2.4

### Depth Camera

- Name: depth_front
- Resolution: 800 × 600
- FOV: 90°

### Semantic Camera

- Name: semantic_front
- Resolution: 800 × 600
- FOV: 90°

### Semantic BEV Camera

- Name: semantic_bev
- Resolution: 512 × 512
- FOV: 90°
- Height: 50 m
- Pitch: -90°

### LiDAR

- Type: ray_cast
- Channels: 32
- Range: 50 m
- Rotation Frequency: 20 Hz
- Points per second: 56000

### GNSS

- Enabled

### IMU

- Enabled

---

## Timing

- FPS: 20
- Fixed delta seconds: 0.05
- Synchronous mode: enabled

---

## BEV Representation

- Size: 512 × 512
- Resolution: 0.2 m/pixel
- Channels:
  - road
  - lane
  - sidewalk

---

## Weather Conditions

- clear_day
- cloudy_day
- rain_day

---

## Maps

Training:
- Town03

Validation:
- Town03

Testing:
- Town05
- Town10HD

---

## Dataset Split

train / val / test

---

## Saved Modalities

- RGB image
- Depth image
- Metric depth (.npy)
- Semantic image
- Semantic raw label
- Semantic BEV
- BEV raw label
- Multi-channel BEV
- LiDAR point cloud
- Pose
- GNSS
- IMU
- Calibration
- Global semantic map

---

## Coordinate System

World coordinates are provided by CARLA.

Global semantic maps use:

u = (x - x_min) / resolution

v = (y - y_min) / resolution

---

## Version

Dataset Version: v1.0