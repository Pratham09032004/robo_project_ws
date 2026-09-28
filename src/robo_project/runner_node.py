#!/usr/bin/env python3

"""
Path follower for the Habitat VineBot.

Follows the /planned_path published by motion_planner using /odom, and
publishes /cmd_vel plus the current target on /runner_current_waypoint.

/map and the map -> odom transform are published by map_server_node, and
odom -> base_footprint by the Habitat bridge. RViz "2D Goal Pose" clicks go
to motion_planner, which replans and publishes a new /planned_path.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path


class RunnerNode(Node):
    def __init__(self):
        super().__init__('runner_node')

        print("\n" + "="*50, flush=True)
        print("🚀 RUNNER NODE INITIALIZING (PATH FOLLOWER, NO ML / NO LOCALIZATION)...", flush=True)
        print("="*50 + "\n", flush=True)

        self.latest_odom_msg = None

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        # Publish the CURRENT target waypoint so a visualizer or other
        # tool can see exactly what runner_node is aiming for right now
        # (not just the full static path from motion_planner).
        self.current_waypoint_pub = self.create_publisher(
            PoseStamped, '/runner_current_waypoint', 10)

        self.cb_group = ReentrantCallbackGroup()

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.get_odom, 10, callback_group=self.cb_group)

        # ── Path following: receives the plan from motion_planner.py ───────────
        path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/planned_path', self.path_callback, path_qos, callback_group=self.cb_group)
        self.path = []
        self.waypoint_idx = 0
        self.robot_yaw = 0.0
        self.path_goal_reached = False
        self.create_timer(0.1, self.path_follow_loop, callback_group=self.cb_group)

    def _publish_cmd_vel(self, fwd: float, ang: float):
        msg = Twist()
        msg.linear.x  = float(fwd)
        msg.angular.z = float(ang)
        self.cmd_vel_pub.publish(msg)

    def get_odom(self, msg: Odometry):
        self.latest_odom_msg = msg
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    # ── Path following ────────────────────────────────────────────────────────

    def path_callback(self, msg: Path):
        if not msg.poses:
            return
        new_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if new_path == self.path:
            return  # ignore republishes of the same path (don't reset progress)
        self.path = new_path
        self.waypoint_idx = 0
        self.path_goal_reached = False
        print(f"Path received from motion_planner: {len(self.path)} waypoints. Following now.", flush=True)

    def path_follow_loop(self):
        if not self.path or self.path_goal_reached:
            return
        odom_msg = self.latest_odom_msg
        if odom_msg is None:
            return

        rx = float(odom_msg.pose.pose.position.x)
        ry = float(odom_msg.pose.pose.position.y)
        ryaw = self.robot_yaw

        if self.waypoint_idx >= len(self.path):
            self._publish_cmd_vel(0.0, 0.0)
            self.path_goal_reached = True
            print("GOAL REACHED! Robot stopped.", flush=True)
            return

        wx, wy = self.path[self.waypoint_idx]

        # Publish current target so it can be visualized live
        wp_msg = PoseStamped()
        wp_msg.header.frame_id = 'map'
        wp_msg.header.stamp = self.get_clock().now().to_msg()
        wp_msg.pose.position.x = wx
        wp_msg.pose.position.y = wy
        wp_msg.pose.orientation.w = 1.0
        self.current_waypoint_pub.publish(wp_msg)

        dist = math.hypot(wx - rx, wy - ry)

        if dist < 0.15:
            self.waypoint_idx += 1
            print(f"Waypoint {self.waypoint_idx}/{len(self.path)} reached.", flush=True)
            return

        angle_to_wp = math.atan2(wy - ry, wx - rx)
        angle_err = self._wrap_angle(angle_to_wp - ryaw)

        # Debug: show exactly what the steering logic is computing,
        # so a direction bug is visible directly in the terminal.
        now = time.time()
        if not hasattr(self, "_last_steer_log") or now - self._last_steer_log > 1.0:
            print(
                f"[steer] robot=({rx:.2f},{ry:.2f}) yaw={math.degrees(ryaw):.0f}° "
                f"-> target=({wx:.2f},{wy:.2f}) dist={dist:.2f}m "
                f"angle_to_wp={math.degrees(angle_to_wp):.0f}° "
                f"angle_err={math.degrees(angle_err):.0f}°",
                flush=True)
            self._last_steer_log = now

        if abs(angle_err) > 0.15:
            ang = 0.5 if angle_err > 0 else -0.5
            self._publish_cmd_vel(0.0, ang)
        else:
            self._publish_cmd_vel(0.3, 0.5 * angle_err)

    @staticmethod
    def _wrap_angle(a):
        while a > math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = RunnerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
