#!/usr/bin/env python3
"""Validate particle-filter localization using dataset replay topics.

This version does NOT call initialize_uniform().
Instead, it waits for the first /odom message and initializes particles around that pose.
"""

from math import atan2, cos, remainder, sin, tau
import time

import numpy as np
import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from robo_project.scripts.basic_types import PoseMeters
from robo_project.scripts.map_handler import MapFrameManager
from robo_project.scripts.particle_filter import ParticleFilter


class ParticleFilterValidationNode(Node):
    def __init__(self):
        super().__init__("particle_filter_validation_node")

        self.declare_parameter("observation_topic", "/local_occupancy_gt")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_topic", "/localization_pose_pf")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("init_xy_noise", 0.10)
        self.declare_parameter("init_yaw_noise", 0.10)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.init_xy_noise = float(self.get_parameter("init_xy_noise").value)
        self.init_yaw_noise = float(self.get_parameter("init_yaw_noise").value)

        self.bridge = CvBridge()
        self.last_odom = None
        self.pf_initialized = False

        self.get_logger().info("ROS publishers and subscribers created")
        self.get_logger().info("Initializing MapFrameManager...")
        self.mfm = MapFrameManager(use_discrete_state_space=False)
        self.get_logger().info("MapFrameManager initialized")

        self.get_logger().info("Creating ParticleFilter object...")
        self.pf = ParticleFilter()
        self.get_logger().info("ParticleFilter object created")

        self.get_logger().info("Setting map frame manager...")
        self.pf.set_map_frame_manager(self.mfm)
        self.get_logger().info("Map frame manager set")

        self.get_logger().info("Skipping initialize_uniform() for dataset replay mode")
        self.get_logger().info("Particles will initialize around first /odom pose")

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("pose_topic").value,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_callback,
            10,
        )

        self.obs_sub = self.create_subscription(
            Image,
            self.get_parameter("observation_topic").value,
            self.observation_callback,
            10,
        )

        self.get_logger().info("Particle-filter validation node started")
        self.get_logger().info(
            f"Listening to odom={self.get_parameter('odom_topic').value}, "
            f"observation={self.get_parameter('observation_topic').value}"
        )

    @staticmethod
    def _yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        return atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _get_particle_array(self):
        if hasattr(self.pf, "particle_set") and self.pf.particle_set is not None:
            return self.pf.particle_set

        self.pf.particle_set = np.zeros((20, 3), dtype=np.float32)
        return self.pf.particle_set

    def _set_uniform_weights_if_present(self, n):
        for name in ["weights", "particle_weights", "weight_set"]:
            if hasattr(self.pf, name):
                setattr(self.pf, name, np.ones(n, dtype=np.float32) / float(n))

    def initialize_particles_around_pose(self, pose):
        particles = self._get_particle_array()
        n = particles.shape[0]

        rng = np.random.default_rng(42)

        particles[:, 0] = pose.x + rng.normal(0.0, self.init_xy_noise, n)
        particles[:, 1] = pose.y + rng.normal(0.0, self.init_xy_noise, n)
        particles[:, 2] = pose.yaw + rng.normal(0.0, self.init_yaw_noise, n)

        self._set_uniform_weights_if_present(n)

        self.pf_initialized = True

        self.get_logger().info(
            f"Particles initialized around first odom: "
            f"x={pose.x:.3f}, y={pose.y:.3f}, yaw={pose.yaw:.3f}, n={n}"
        )

    def estimate_from_particles(self):
        particles = self._get_particle_array()

        x = float(np.mean(particles[:, 0]))
        y = float(np.mean(particles[:, 1]))
        yaw = float(
            atan2(
                np.mean(np.sin(particles[:, 2])),
                np.mean(np.cos(particles[:, 2])),
            )
        )

        return PoseMeters(x, y, yaw)

    def odom_callback(self, msg):
        current = PoseMeters(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self._yaw_from_odom(msg),
        )

        if not self.pf_initialized:
            self.initialize_particles_around_pose(current)
            self.last_odom = current
            return

        if self.last_odom is not None:
            dx = current.x - self.last_odom.x
            dy = current.y - self.last_odom.y

            fwd = cos(self.last_odom.yaw) * dx + sin(self.last_odom.yaw) * dy
            dyaw = remainder(current.yaw - self.last_odom.yaw, tau)

            try:
                self.pf.propagate_particles(fwd, dyaw)
            except Exception as exc:
                self.get_logger().error(f"Motion update failed: {exc}")

        self.last_odom = current

    def observation_callback(self, msg):
        self.get_logger().info("Observation callback started")

        if not self.pf_initialized:
            self.get_logger().warn("Observation received before first odom. Waiting for odom.")
            return

        try:
            grid = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            grid = np.asarray(grid, dtype=np.float32)

            self.get_logger().info(f"Grid received: shape={grid.shape}, dtype={grid.dtype}")

            # Replay node publishes:
            # free = 255, occupied = 0, unknown = 127
            # Convert to float approximately: free=1, occupied=0, unknown=0.5
            grid = grid / 255.0

            expected_h = int(self.mfm.obs_height_px)
            expected_w = int(self.mfm.obs_width_px)
            if grid.shape != (expected_h, expected_w):
                grid = cv2.resize(
                    grid,
                    (expected_w, expected_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                self.get_logger().info(
                    f"Grid resized for PF: shape={grid.shape}, dtype={grid.dtype}"
                )

            self.get_logger().info("DEBUG 1: before particle filter update")
            t0 = time.time()

            # Real particle-filter correction:
            # Compare received local occupancy grid with expected map observations.
            estimate = self.pf.update_with_observation(grid)

            dt = time.time() - t0
            self.get_logger().info(f"DEBUG 2: after particle filter update, dt={dt:.3f}s")

            try:
                self.pf.resample()
                self.get_logger().info("DEBUG 3: after resample")
            except Exception as exc:
                self.get_logger().warn(f"Resample failed: {exc}")

            if estimate is None:
                estimate = self.estimate_from_particles()

            output = PoseStamped()
            output.header.stamp = self.get_clock().now().to_msg()
            output.header.frame_id = self.frame_id

            output.pose.position.x = float(estimate.x)
            output.pose.position.y = float(estimate.y)
            output.pose.position.z = 0.0

            output.pose.orientation.x = 0.0
            output.pose.orientation.y = 0.0
            output.pose.orientation.z = sin(float(estimate.yaw) / 2.0)
            output.pose.orientation.w = cos(float(estimate.yaw) / 2.0)

            self.pose_pub.publish(output)

            self.get_logger().info(
                f"Published PF pose: x={estimate.x:.3f}, "
                f"y={estimate.y:.3f}, yaw={estimate.yaw:.3f}"
            )
            self.get_logger().info("Observation callback finished")

        except Exception as exc:
            self.get_logger().error(f"Observation callback failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterValidationNode()

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
