#!/usr/bin/env python3

"""
Interface to parse sensor data from the locobot.
Migrated from ROS 1 to ROS 2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2  # ROS 2: was sensor_msgs.point_cloud2

import numpy as np
import yaml, os, cv2
from cv_bridge import CvBridge
try:
    from bresenham import bresenham
except ImportError:
    def bresenham(x0, y0, x1, y1):
        """Integer cells on the line (x0, y0) -> (x1, y1); same output as the pip 'bresenham' package."""
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
from typing import Tuple

from robo_project.scripts.config_loader import load_config

# ROS 2: use rclpy logger
import rclpy.logging
_logger = rclpy.logging.get_logger('vinebot_interface')

g_cv_bridge = CvBridge()

g_pointcloud_msg = None
g_lidar_local_occ_meas = None
g_lidar_detects_robot_facing_wall: bool = False
g_depth_local_occ_meas = None
g_pointcloud_local_occ_meas = None

# Config params.
g_local_occ_size: int = None
g_local_occ_resolution: float = None


def show_images(event=None):
    """
    When running this as a standalone node, display result images.
    """
    if g_lidar_local_occ_meas is not None:
        cv2.namedWindow("LiDAR -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
        cv2.imshow("LiDAR -> local occ meas (front = right)", g_lidar_local_occ_meas)

    if g_depth_local_occ_meas is not None:
        cv2.namedWindow("Depth Img -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth Img -> local occ meas (front = right)", g_depth_local_occ_meas)

    global g_pointcloud_msg
    if g_pointcloud_msg is not None:
        pc = g_pointcloud_msg
        g_pointcloud_msg = None
        get_local_occ_from_pointcloud(pc)
        cv2.namedWindow("Pointcloud -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
        cv2.imshow("Pointcloud -> local occ meas (front = right)", g_pointcloud_local_occ_meas)

    cv2.waitKey(100)


def get_local_occ_from_lidar(msg: LaserScan):
    """
    Convert a LiDAR scan into a pseudo-local-occupancy measurement.
    @param msg - LaserScan from the 360 degree planar LiDAR.
    """
    local_occ_meas = np.ones((g_local_occ_size, g_local_occ_size))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = msg.range_max / g_local_occ_resolution

    for i in range(len(msg.ranges)):
        if msg.ranges[i] < msg.range_min or msg.ranges[i] > msg.range_max:
            continue
        angle = msg.angle_min + i * msg.angle_increment
        dist_px = msg.ranges[i] / g_local_occ_resolution
        r_hit = center_r + int(dist_px * np.sin(angle))
        c_hit = center_c - int(dist_px * np.cos(angle))
        r_max_range = center_r + int(max_range_px * np.sin(angle))
        c_max_range = center_c - int(max_range_px * np.cos(angle))
        for cell in bresenham(r_hit, c_hit, r_max_range, c_max_range):
            r = cell[0]; c = cell[1]
            if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
                local_occ_meas[r, c] = 0
            else:
                break

    # Check if the area in front of the robot is occluded.
    global g_lidar_detects_robot_facing_wall
    front_cell_block = local_occ_meas[
        local_occ_meas.shape[0] // 3:2 * local_occ_meas.shape[0] // 3,
        2 * local_occ_meas.shape[0] // 3:
    ]
    front_cell_mean = np.mean(front_cell_block)
    g_lidar_detects_robot_facing_wall = front_cell_mean <= 0.75

    global g_lidar_local_occ_meas
    g_lidar_local_occ_meas = local_occ_meas.copy()


def get_local_occ_from_depth(msg: Image):
    """
    Process a depth image into a local occupancy measurement.
    @param msg - Raw rectified depth image from the RealSense.
    @return local occupancy grid (only EAST quadrant is meaningful due to FOV).
    """
    depth_img = g_cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').copy()

    local_occ_meas = np.ones((g_local_occ_size, g_local_occ_size))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = g_local_occ_size
    range_max = max_range_px * g_local_occ_resolution

    rs_range_min = 0.1
    rs_range_max = 5.0

    ratio_to_use = 1.0
    depth_img_top_half = depth_img[:int(ratio_to_use * depth_img.shape[0]), :]
    depth_img_top_half[depth_img_top_half < rs_range_min] = np.mean(
        depth_img_top_half[depth_img_top_half >= rs_range_min])
    flat_depth_meas = np.mean(depth_img_top_half, axis=0)

    depth_fov = np.deg2rad(90)
    half_depth_fov = depth_fov / 2
    num_rays = flat_depth_meas.shape[0]
    dtheta = depth_fov / num_rays
    angles = [-half_depth_fov + dtheta * i for i in range(num_rays)]
    ignore_edge_regions_area = np.deg2rad(0.0)
    angle_upper_mag_to_keep = half_depth_fov - ignore_edge_regions_area

    for i in range(num_rays):
        angle = angles[i]
        if angle < -angle_upper_mag_to_keep or angle > angle_upper_mag_to_keep:
            continue
        depth = flat_depth_meas[i] * 0.001  # mm to meters
        if depth < rs_range_min or depth > range_max:
            continue
        dist_px = depth / g_local_occ_resolution
        r_hit = center_r + int(dist_px * np.sin(angle))
        c_hit = center_c + int(dist_px * np.cos(angle))
        r_max_range = center_r + int(max_range_px * np.sin(angle))
        c_max_range = center_c + int(max_range_px * np.cos(angle))
        for cell in bresenham(r_hit, c_hit, r_max_range, c_max_range):
            r = cell[0]; c = cell[1]
            if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
                local_occ_meas[r, c] = 0
            else:
                break

    global g_depth_local_occ_meas
    g_depth_local_occ_meas = local_occ_meas.copy()
    return local_occ_meas


def get_local_occ_from_pointcloud(msg: PointCloud2):
    """
    Process a pointcloud message into a local occupancy measurement.
    @param msg - PointCloud2 from the RealSense depth sensor.
    @return local occupancy grid (only EAST quadrant is meaningful due to FOV).
    """
    # ROS 2: sensor_msgs_py.point_cloud2 replaces sensor_msgs.point_cloud2
    gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))

    local_occ_meas = np.ones((g_local_occ_size, g_local_occ_size))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = g_local_occ_size

    for pt in gen:
        if pt[0] > 5 or pt[1] > 5 or pt[2] > 5:
            continue
        if pt[2] < 0.001:
            continue
        dc = pt[2] / g_local_occ_resolution
        dr = pt[0] / g_local_occ_resolution
        r_hit = center_r + dr
        c_hit = center_c + dc

        if r_hit < 0 or c_hit < 0 or r_hit >= g_local_occ_size or c_hit >= g_local_occ_size:
            continue

        angle = np.arctan2(dr, dc)
        r_max_range = center_r + max_range_px * np.sin(angle)
        c_max_range = center_c + max_range_px * np.cos(angle)
        for cell in bresenham(int(r_hit), int(c_hit), int(r_max_range), int(c_max_range)):
            r = cell[0]; c = cell[1]
            if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
                local_occ_meas[r, c] = 0
            else:
                break

    global g_pointcloud_local_occ_meas
    g_pointcloud_local_occ_meas = local_occ_meas.copy()
    return local_occ_meas


def get_pointcloud_msg(msg: PointCloud2):
    """
    Cache a pointcloud message to be processed when needed.
    """
    global g_pointcloud_msg
    g_pointcloud_msg = msg


def read_params():
    """
    Read configuration params from the yaml.
    """
    config = load_config()
    global g_local_occ_size, g_local_occ_resolution
    g_local_occ_size = config["lidar"]["local_occ_size"]
    g_local_occ_resolution = config["lidar"]["local_occ_resolution"]


class LocobotInterfaceNode(Node):
    """
    Standalone ROS 2 node for the locobot interface
    (ros2 run robo_project vinebot_interface).
    """

    def __init__(self):
        super().__init__('interface_node')

        # Subscribers. /camera/front/image_raw carries Images, not point clouds;
        # the Livox lidar point cloud is bridged from Gazebo to /points.
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("pointcloud_topic", "/points")
        self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value, get_local_occ_from_lidar, 1)
        self.create_subscription(
            PointCloud2, self.get_parameter("pointcloud_topic").value, get_pointcloud_msg, 1)

        # Timer for displaying images when running standalone
        # ROS 2: create_timer(period_seconds, callback)
        self.create_timer(0.1, show_images)

        self.get_logger().info("LocobotInterfaceNode started.")


def main(args=None):
    rclpy.init(args=args)
    read_params()
    try:
        node = LocobotInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
