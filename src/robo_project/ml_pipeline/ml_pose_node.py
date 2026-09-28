#!/usr/bin/env python3

import os

import rclpy

from rclpy.node import Node

from geometry_msgs.msg import Point

# Change line 11 to:
from robo_project.ml_pipeline.infer import predict_pose


class MLPoseNode(Node):

    def __init__(self):

        super().__init__("ml_estimated_pose_node")

        self.pose_pub = self.create_publisher(
            Point,
            "/ml_estimated_pose",
            10
        )

        self.declare_parameter(
            "image_folder",
            os.path.expanduser(
                "~/robo_project_ws/src/dataset/front"
            )
        )

        self.image_folder = os.path.expanduser(self.get_parameter(
            "image_folder"
        ).value)

        self.image_files = sorted(
            os.listdir(self.image_folder)
        )

        self.image_index = 0

        self.timer = self.create_timer(
            1.0,
            self.run_navigation
        )

        self.get_logger().info(
            "ML Pose Node Started"
        )

    def run_navigation(self):

        if self.image_index >= len(self.image_files):

            self.get_logger().info(
                "Finished all dataset images"
            )

            rclpy.shutdown()

            return

        image_name = self.image_files[
            self.image_index
        ]

        image_path = os.path.join(
            self.image_folder,
            image_name
        )

        x, z = predict_pose(
            image_path
        )

        self.get_logger().info(
            f"Image: {image_name}"
        )

        self.get_logger().info(
            f"Predicted Position -> X: {x:.2f}, Z: {z:.2f}"
        )

        pose_msg = Point()

        pose_msg.x = float(x)
        pose_msg.y = float(z)
        pose_msg.z = 0.0

        self.pose_pub.publish(
            pose_msg
        )

        self.get_logger().info(
            f"Published Pose -> X: {x:.2f}, Z: {z:.2f}"
        )

        self.image_index += 1


def main(args=None):

    rclpy.init(args=args)

    node = MLPoseNode()

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
