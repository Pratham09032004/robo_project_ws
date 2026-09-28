#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from geometry_msgs.msg import Point

from nav_msgs.msg import Odometry

from math import atan2


class PoseMonitorNode(Node):

    def __init__(self):

        super().__init__("pose_monitor_node")

        self.ml_x = None
        self.ml_z = None

        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None

        self.create_subscription(
            Point,
            "/ml_estimated_pose",
            self.ml_callback,
            10
        )

        # /odom is published by the Habitat bridge; use odom_topic:=/diff_cont/odom in Gazebo.
        self.declare_parameter("odom_topic", "/odom")

        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(
            1.0,
            self.print_status
        )

        self.get_logger().info(
            "Pose Monitor Node Started"
        )

    def ml_callback(self, msg):

        self.ml_x = msg.x
        self.ml_z = msg.y

    def odom_callback(self, msg):

        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        self.odom_yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def print_status(self):

        print("\n====================")

        print("ML Pose")

        print(
            f"X: {self.ml_x}, Z: {self.ml_z}"
        )

        print("\nOdometry")

        print(
            f"X: {self.odom_x}, Y: {self.odom_y}, Yaw: {self.odom_yaw}"
        )

        print("====================\n")


def main(args=None):

    rclpy.init(args=args)

    node = PoseMonitorNode()

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
