# KEN Robots LiDAR Perception Pipeline

This project implements a LiDAR-based perception pipeline using **ROS2 Humble**, **Gazebo**, and **RViz**.

## Features
- Gazebo LiDAR simulation
- ROS2 ↔ Gazebo point cloud bridge
- Point cloud filtering
- Voxel downsampling
- Ground removal
- Obstacle clustering
- Obstacle intelligence:
  - nearest obstacle
  - collision radius
  - safe corridor width
  - safe direction
- OctoMap integration
- RViz visualization
- Simulation testing in custom Gazebo worlds

## Topics
- `/lidar/points`
- `/lidar/points_filtered`
- `/lidar/points_voxel`
- `/lidar/points_noground`
- `/lidar/clusters_cloud`
- `/lidar/clusters_markers`
- `/nearest_obstacle`
- `/obstacle_map`
- `/safe_direction`

## Run
```bash
source /opt/ros/humble/setup.bash
cd ~/ken_robots_ws
colcon build --symlink-install
source install/setup.bash
ros2 run lidar_processing filter_node
