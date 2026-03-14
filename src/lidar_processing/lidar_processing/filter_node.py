#!/usr/bin/env python3
import random
from typing import Dict, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray


def make_pointcloud2_xyz(frame_id: str, stamp, pts_xyz: np.ndarray) -> PointCloud2:
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = int(pts_xyz.shape[0])
    msg.is_bigendian = False
    msg.is_dense = False

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width

    if msg.width == 0:
        msg.data = b""
        return msg

    msg.data = np.asarray(pts_xyz, dtype=np.float32).tobytes()
    return msg


class LidarFilterNode(Node):
    def __init__(self):
        super().__init__('lidar_filter_node')
        self.frame_count = 0
        self.qos = QoSProfile(depth=10)
        self.qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.qos.history = QoSHistoryPolicy.KEEP_LAST
        
        self.declare_parameter('input_topic', '/lidar/points')
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('min_z', -1.0)
        self.declare_parameter('max_z', 2.0)
        self.declare_parameter('voxel_size', 0.15)
        self.declare_parameter('ransac_iters', 50)
        self.declare_parameter('plane_dist_thresh', 0.08)
        self.declare_parameter('min_plane_inliers', 20)
        self.declare_parameter('plane_upright_cos', 0.85)
        self.declare_parameter('cluster_min_points', 3)
        self.declare_parameter('cluster_max_points', 1000)

        input_topic = self.get_parameter('input_topic').value

        self.sub = self.create_subscription(PointCloud2, input_topic, self.cb, self.qos)

        self.pub_filtered = self.create_publisher(PointCloud2, '/lidar/points_filtered', self.qos)
        self.pub_voxel = self.create_publisher(PointCloud2, '/lidar/points_voxel', self.qos)
        self.pub_noground = self.create_publisher(PointCloud2, '/lidar/points_noground', self.qos)
        self.pub_clusters = self.create_publisher(PointCloud2, '/lidar/clusters_cloud', self.qos)
        self.pub_markers = self.create_publisher(MarkerArray, '/lidar/clusters_markers', self.qos)

        self.pub_obstacle_map = self.create_publisher(PointCloud2, '/obstacle_map', self.qos)
        self.pub_nearest = self.create_publisher(Float32, '/nearest_obstacle', 10)
        self.pub_safe_direction = self.create_publisher(Vector3, '/safe_direction', 10)
        self.pub_collision_radius = self.create_publisher(Float32, '/collision_radius', 10)
        self.pub_safe_corridor_width = self.create_publisher(Float32, '/safe_corridor_width', 10)

        self.get_logger().info(f"Phase6 node running. Sub: {input_topic}")

    def cb(self, msg: PointCloud2):
        pts = self.read_points_xyz(msg)
        if pts.shape[0] == 0:
            self.publish_empty(msg)
            return

        pts_f = self.filter_passthrough_and_range(pts)
        self.pub_filtered.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, pts_f))

        if pts_f.shape[0] == 0:
            self.publish_empty(msg, already_filtered=True)
            return

        vox = self.voxel_downsample(pts_f)
        self.pub_voxel.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, vox))

        if vox.shape[0] == 0:
            self.publish_empty(msg, already_filtered=True)
            return

        noground = self.remove_ground_ransac(vox)
        self.pub_noground.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, noground))

        clusters = self.cluster_voxels(noground)
        clusters_pts = np.vstack(clusters) if len(clusters) > 0 else np.zeros((0, 3), dtype=np.float32)

        self.pub_clusters.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, clusters_pts))
        self.pub_obstacle_map.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, clusters_pts))
        self.pub_markers.publish(self.make_cluster_markers(msg, clusters))

        nearest = self.compute_nearest_distance(clusters)
        self.pub_nearest.publish(Float32(data=float(nearest)))

        collision_radius = self.compute_collision_radius(clusters)
        self.pub_collision_radius.publish(Float32(data=collision_radius))

        safe_corridor_width = self.compute_safe_corridor_width(noground)
        self.pub_safe_corridor_width.publish(Float32(data=safe_corridor_width))

        safe_direction = self.compute_safe_direction(noground)
        self.pub_safe_direction.publish(safe_direction)
        
        self.frame_count += 1
        if self.frame_count % 20 == 0:
           self.get_logger().info(
               f"pts={pts.shape[0]}, filtered={pts_f.shape[0]}, voxel={vox.shape[0]}, noground={noground.shape[0]}, clusters={len(clusters)}"
            )
           
    def read_points_xyz(self, msg: PointCloud2) -> np.ndarray:
        it = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts_list = list(it)

        if len(pts_list) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        arr = np.asarray(pts_list)

        if hasattr(arr.dtype, "fields") and arr.dtype.fields is not None:
            x = arr['x'].astype(np.float32)
            y = arr['y'].astype(np.float32)
            z = arr['z'].astype(np.float32)
            return np.stack([x, y, z], axis=1)

        return np.asarray(arr, dtype=np.float32).reshape(-1, 3)

    def publish_empty(self, msg: PointCloud2, already_filtered: bool = False):
        empty = np.zeros((0, 3), dtype=np.float32)

        if not already_filtered:
            self.pub_filtered.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, empty))

        self.pub_voxel.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, empty))
        
        if vox.shape[0] < 3:
            self.publish_empty(msg, already_filtered=True)
            return

        self.pub_noground.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, empty))
        
        if noground.shape[0] < 3:
            self.pub_clusters.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, np.zeros((0, 3), dtype=np.float32)))
            self.pub_obstacle_map.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, np.zeros((0, 3), dtype=np.float32)))
            self.pub_markers.publish(MarkerArray(markers=[]))
            self.pub_nearest.publish(Float32(data=float('inf')))
            self.pub_collision_radius.publish(Float32(data=0.0))
            self.pub_safe_corridor_width.publish(Float32(data=10.0))
            forward = Vector3()
            forward.x = 1.0
            self.pub_safe_direction.publish(forward)
            return
        
        self.pub_clusters.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, empty))
        self.pub_obstacle_map.publish(make_pointcloud2_xyz(msg.header.frame_id, msg.header.stamp, empty))
        self.pub_markers.publish(MarkerArray(markers=[]))
        self.pub_nearest.publish(Float32(data=float('inf')))
        self.pub_collision_radius.publish(Float32(data=0.0))
        self.pub_safe_corridor_width.publish(Float32(data=10.0))
        forward = Vector3()
        forward.x = 1.0
        forward.y = 0.0
        forward.z = 0.0
        self.pub_safe_direction.publish(forward)

    def filter_passthrough_and_range(self, pts: np.ndarray) -> np.ndarray:
        max_range = float(self.get_parameter('max_range').value)
        min_z = float(self.get_parameter('min_z').value)
        max_z = float(self.get_parameter('max_z').value)

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y)

        mask = (r <= max_range) & (z >= min_z) & (z <= max_z)
        return pts[mask]

    def voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        vs = float(self.get_parameter('voxel_size').value)
        if pts.shape[0] == 0:
            return pts

        q = np.floor(pts / vs).astype(np.int32)
        _, idx = np.unique(q, axis=0, return_index=True)
        return pts[idx]

    def remove_ground_ransac(self, pts: np.ndarray) -> np.ndarray:
        iters = int(self.get_parameter('ransac_iters').value)
        dist_th = float(self.get_parameter('plane_dist_thresh').value)
        min_inliers = int(self.get_parameter('min_plane_inliers').value)
        upright_cos = float(self.get_parameter('plane_upright_cos').value)

        n = pts.shape[0]
        if n < 30:
            return pts

        best_inliers = None
        best_count = 0

        for _ in range(iters):
            i1, i2, i3 = random.sample(range(n), 3)
            p1, p2, p3 = pts[i1], pts[i2], pts[i3]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-6:
                continue

            normal = normal / norm

            if abs(normal[2]) < upright_cos:
                continue

            d = -np.dot(normal, p1)
            dist = np.abs(pts @ normal + d)
            inliers = dist < dist_th
            count = int(np.sum(inliers))

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < min_inliers:
            return pts

        return pts[~best_inliers]

    def cluster_voxels(self, pts: np.ndarray) -> List[np.ndarray]:
        if pts.shape[0] == 0:
            return []

        vs = float(self.get_parameter('voxel_size').value)
        min_pts = int(self.get_parameter('cluster_min_points').value)
        max_pts = int(self.get_parameter('cluster_max_points').value)

        q = np.floor(pts / vs).astype(np.int32)

        voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
        for i, key in enumerate(map(tuple, q)):
            voxel_map.setdefault(key, []).append(i)

        voxels = list(voxel_map.keys())
        voxel_set = set(voxels)

        neigh = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        visited = set()
        clusters: List[np.ndarray] = []

        for v in voxels:
            if v in visited:
                continue

            stack = [v]
            visited.add(v)
            comp_voxels = []

            while stack:
                cur = stack.pop()
                comp_voxels.append(cur)

                cx, cy, cz = cur
                for dx, dy, dz in neigh:
                    nb = (cx + dx, cy + dy, cz + dz)
                    if nb in voxel_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

            comp_indices = []
            for vv in comp_voxels:
                comp_indices.extend(voxel_map[vv])

            if len(comp_indices) < min_pts:
                continue
            if len(comp_indices) > max_pts:
                comp_indices = comp_indices[:max_pts]

            clusters.append(pts[np.array(comp_indices, dtype=np.int32)])

        return clusters

    def compute_nearest_distance(self, clusters: List[np.ndarray]) -> float:
        if len(clusters) == 0:
            return float('inf')

        best = float('inf')
        for c in clusters:
            if c.shape[0] == 0:
                continue
            d = np.sqrt(np.sum(c[:, 0:2] ** 2, axis=1))
            m = float(np.min(d))
            if m < best:
                best = m
        return best

    def compute_collision_radius(self, clusters: List[np.ndarray]) -> float:
        if len(clusters) == 0:
            return 0.0

        best_cluster = None
        best_dist = float('inf')

        for c in clusters:
            if c.shape[0] == 0:
                continue
            d = np.sqrt(np.sum(c[:, 0:2] ** 2, axis=1))
            m = float(np.min(d))
            if m < best_dist:
                best_dist = m
                best_cluster = c

        if best_cluster is None:
            return 0.0

        min_xyz = np.min(best_cluster, axis=0)
        max_xyz = np.max(best_cluster, axis=0)
        size = max_xyz - min_xyz

        return float(max(size[0], size[1]) / 2.0)

    def compute_safe_corridor_width(self, pts: np.ndarray) -> float:
        if pts.shape[0] == 0:
            return 0.0

        front_pts = pts[(pts[:, 0] > 0.0) & (pts[:, 0] < 2.5)]

        if front_pts.shape[0] == 0:
            return 10.0

        left_pts = front_pts[front_pts[:, 1] > 0.0]
        right_pts = front_pts[front_pts[:, 1] < 0.0]

        if left_pts.shape[0] == 0 or right_pts.shape[0] == 0:
            return 10.0

        left_inner = np.min(left_pts[:, 1])
        right_inner = np.max(right_pts[:, 1])

        corridor_width = left_inner - right_inner
        return float(max(corridor_width, 0.0))

    def compute_safe_direction(self, pts: np.ndarray) -> Vector3:
        direction = Vector3()

        if pts.shape[0] == 0:
            direction.x = 1.0
            direction.y = 0.0
            direction.z = 0.0
            return direction

        front_range = 2.5
        front_pts = pts[(pts[:, 0] > 0.0) & (pts[:, 0] < front_range)]

        left_pts = front_pts[front_pts[:, 1] > 0.3]
        center_pts = front_pts[np.abs(front_pts[:, 1]) <= 0.3]
        right_pts = front_pts[front_pts[:, 1] < -0.3]

        def nearest_distance(arr):
            if arr.shape[0] == 0:
                return float('inf')
            return float(np.min(np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)))

        left_dist = nearest_distance(left_pts)
        center_dist = nearest_distance(center_pts)
        right_dist = nearest_distance(right_pts)

        if center_dist > 1.5:
            direction.x = 1.0
            direction.y = 0.0
            direction.z = 0.0
        elif left_dist > right_dist and left_dist > 0.8:
            direction.x = 0.0
            direction.y = 1.0
            direction.z = 0.0
        elif right_dist >= left_dist and right_dist > 0.8:
            direction.x = 0.0
            direction.y = -1.0
            direction.z = 0.0
        else:
            direction.x = -1.0
            direction.y = 0.0
            direction.z = 0.0

        return direction

    def make_cluster_markers(self, msg: PointCloud2, clusters: List[np.ndarray]) -> MarkerArray:
        arr = MarkerArray()

        clear = Marker()
        clear.header = msg.header
        clear.ns = "clusters_bbox"
        clear.id = 0
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        mid = 0
        for c in clusters:
            if c.shape[0] == 0:
                continue

            min_xyz = np.min(c, axis=0)
            max_xyz = np.max(c, axis=0)
            center = (min_xyz + max_xyz) / 2.0
            size = (max_xyz - min_xyz)

            m = Marker()
            m.header = msg.header
            m.ns = "clusters_bbox"
            m.id = mid
            mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.w = 1.0

            m.scale.x = float(max(size[0], 0.05))
            m.scale.y = float(max(size[1], 0.05))
            m.scale.z = float(max(size[2], 0.05))

            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.6

            arr.markers.append(m)

        return arr


def main(args=None):
    rclpy.init(args=args)
    node = LidarFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()