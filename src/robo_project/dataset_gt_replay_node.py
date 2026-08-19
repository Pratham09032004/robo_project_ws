#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def yaw_to_quat(yaw):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


class DatasetGTReplayNode(Node):
    def __init__(self):
        super().__init__("dataset_gt_replay_node")

        self.declare_parameter(
            "dataset_path",
            str(Path.home() / "robo_project_ws" / "dataset" / "cmn_dataset_delton"),
        )
        self.declare_parameter("trajectory_id", "trajectory_01")
        self.declare_parameter("publish_period", 0.2)

        self.dataset_path = Path(self.get_parameter("dataset_path").value)
        self.trajectory_id = self.get_parameter("trajectory_id").value
        self.publish_period = float(self.get_parameter("publish_period").value)

        self.bridge = CvBridge()

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.occ_pub = self.create_publisher(Image, "/local_occupancy_gt", 10)

        self.samples = self.load_samples()
        self.index = 0

        self.timer = self.create_timer(self.publish_period, self.timer_callback)

        self.get_logger().info(
            f"Loaded {len(self.samples)} samples from {self.trajectory_id}"
        )

    def load_samples(self):
        samples = []

        for sample_dir in sorted(self.dataset_path.glob("sample_*")):
            meta_path = sample_dir / "metadata.json"
            occ_path = sample_dir / "local_occupancy_gt.npy"

            if not meta_path.exists() or not occ_path.exists():
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("trajectory_id") == self.trajectory_id:
                samples.append((sample_dir, meta))

        if not samples:
            raise RuntimeError(
                f"No samples found for trajectory_id={self.trajectory_id} in dataset_path={self.dataset_path}. Copy the dataset to ~/robo_project_ws/dataset/cmn_dataset_delton or pass dataset_path:=/your/path"
            )

        samples.sort(key=lambda x: x[1].get("frame_id", ""))
        return samples

    def timer_callback(self):
        if self.index >= len(self.samples):
            self.get_logger().info("Restarting dataset trajectory.")
            self.index = 0
            return

        sample_dir, meta = self.samples[self.index]
        stamp = self.get_clock().now().to_msg()

        odom_data = meta.get("odometry_delta", {})
        ros_pose = meta.get("ros_map_pose", {})

        x = float(
            odom_data.get(
                "odom_x",
                ros_pose.get("x", meta["position_m"]["x"]),
            )
        )
        y = float(
            odom_data.get(
                "odom_y",
                ros_pose.get("y", meta["position_m"]["z"]),
            )
        )
        yaw = float(
            odom_data.get(
                "odom_yaw",
                ros_pose.get("yaw", meta.get("yaw_rad", 0.0)),
            )
        )

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        self.odom_pub.publish(odom)

        occ = np.load(sample_dir / "local_occupancy_gt.npy").astype(np.float32)

        occ_img = np.zeros_like(occ, dtype=np.uint8)
        occ_img[occ == 1] = 255
        occ_img[occ == 0] = 0
        occ_img[occ == -1] = 127

        msg = self.bridge.cv2_to_imgmsg(occ_img, encoding="mono8")
        msg.header.stamp = stamp
        msg.header.frame_id = "base_footprint"

        self.occ_pub.publish(msg)

        self.get_logger().info(
            f"Published {sample_dir.name} | x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DatasetGTReplayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
