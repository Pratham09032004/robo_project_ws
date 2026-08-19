#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from tf2_ros import TransformBroadcaster

import yaml, os, time, math
from ament_index_python.packages import get_package_share_directory
import numpy as np

from robo_project.scripts.map_handler import MapFrameManager
from robo_project.scripts.basic_types import PoseMeters, PosePixels

class RunnerNode(Node):
    def __init__(self):
        super().__init__('runner_node')

        print("\n" + "="*50, flush=True)
        print("🚀 RUNNER NODE INITIALIZING (NO ML / NO LOCALIZATION)...", flush=True)
        print("="*50 + "\n", flush=True)

        self.mfm = None

        # QoS perfectly matched to RViz settings
        map_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_odom_msg_for_tf = None
        self.rviz_tf_timer = self.create_timer(0.05, self._publish_stable_rviz_tf_timer)

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
        self.create_subscription(PoseStamped, '/goal_pose', self.rviz_goal_callback, 10, callback_group=self.cb_group)

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

        self.setup_complete = False
        self.map_published_once = False

        self.startup_timer = self.create_timer(1.0, self.async_setup, callback_group=self.cb_group)
        self.create_timer(1.0, self.publish_map, callback_group=self.cb_group)

    def async_setup(self):
        if self.setup_complete: return
        self.startup_timer.cancel()

        print("⏳ Loading map in the background...", flush=True)
        try:
            self.mfm = MapFrameManager(use_discrete_state_space=True)
            self.setup_complete = True
            print("✅ BACKGROUND LOAD COMPLETE! Map is now actively broadcasting.", flush=True)
        except Exception as e:
            print(f"❌ ERROR DURING STARTUP: {e}", flush=True)

    def _publish_cmd_vel(self, fwd: float, ang: float):
        msg = Twist()
        msg.linear.x  = float(fwd)
        msg.angular.z = float(ang)
        self.cmd_vel_pub.publish(msg)

    def publish(self, twist_msg):
        is_turn = abs(twist_msg.angular.z) > 0.01
        is_move = abs(twist_msg.linear.x) > 0.01

        if is_turn:
            print("🤖 AI COMMAND: Perfect 90-degree turn.", flush=True)
            cmd = Twist()
            cmd.angular.z = math.copysign(0.5, twist_msg.angular.z)
            for _ in range(10):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

        elif is_move:
            print("🤖 AI COMMAND: Perfect grid step forward.", flush=True)
            cmd = Twist()
            cmd.linear.x = math.copysign(0.5, twist_msg.linear.x)
            for _ in range(10):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.5)

    def publish_map(self):
        if not self.setup_complete or self.mfm is None:
            return

        grid_map = self.mfm.map
        if grid_map is None: return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = t.header.stamp
        resolution = 0.02
        msg.info.resolution = resolution
        msg.info.height = int(grid_map.shape[0])
        msg.info.width = int(grid_map.shape[1])
        msg.info.origin.position.x = float(-(grid_map.shape[1] / 2.0) * resolution)
        msg.info.origin.position.y = float(-(grid_map.shape[0] / 2.0) * resolution)
        msg.info.origin.orientation.w = 1.0

        msg.data = np.where(grid_map == 1, 100, 0).astype(np.int8).flatten().tolist()
        self.map_pub.publish(msg)

        if not self.map_published_once:
            print("🗺️ MAP BROADCAST SUCCESSFUL! Check RViz now.", flush=True)
            self.map_published_once = True

    def _nearest_free_pose_px(self, pose_px, max_radius: int = 120):
        """
        Snap any pixel pose to the nearest valid free cell on the hand-drawn map.
        Internal map convention: 1 = free, 0 = obstacle.
        """
        if pose_px is None:
            return None
        if self.mfm is None:
            return pose_px

        m = self.mfm.map_with_border
        H, W = m.shape

        r0 = int(round(pose_px.r))
        c0 = int(round(pose_px.c))
        yaw = float(getattr(pose_px, "yaw", 0.0))

        r0 = max(0, min(H - 1, r0))
        c0 = max(0, min(W - 1, c0))

        if m[r0, c0] == 1:
            return PosePixels(r0, c0, yaw)

        best = None
        best_d2 = None

        for rad in range(1, max_radius + 1):
            for dr in range(-rad, rad + 1):
                for dc in (-rad, rad):
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < H and 0 <= c < W and m[r, c] == 1:
                        d2 = dr * dr + dc * dc
                        if best is None or d2 < best_d2:
                            best = (r, c)
                            best_d2 = d2

            for dc in range(-rad + 1, rad):
                for dr in (-rad, rad):
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < H and 0 <= c < W and m[r, c] == 1:
                        d2 = dr * dr + dc * dc
                        if best is None or d2 < best_d2:
                            best = (r, c)
                            best_d2 = d2

            if best is not None:
                r, c = best
                return PosePixels(r, c, yaw)

        return PosePixels(r0, c0, yaw)

    def rviz_goal_callback(self, msg):
        self.goal_received = True
        print("🎯 Goal received from RViz.", flush=True)
        print("Goal pose received on /goal_pose", flush=True)

        if not self.setup_complete or self.mfm is None:
            print("❌ Goal ignored: Map Frame Manager not ready.", flush=True)
            return

        pose_m = PoseMeters(float(msg.pose.position.x), float(msg.pose.position.y), 0.0)
        raw_goal_px = self.mfm.transform_pose_m_to_px(pose_m)
        goal_px = self._nearest_free_pose_px(raw_goal_px, max_radius=150)

        if goal_px is None:
            print("❌ Goal ignored: could not convert goal pose to map pixel.", flush=True)
            return

        print(
            f"🎯 Goal mapped: raw_px=({int(raw_goal_px.r)}, {int(raw_goal_px.c)}) "
            f"→ free_px=({int(goal_px.r)}, {int(goal_px.c)})",
            flush=True
        )

        self.current_goal_px = goal_px
        self.goal_received = True
        print("✅ Goal set from RViz goal_pose", flush=True)

    def _publish_stable_rviz_tf_timer(self):
        """
        Permanent RViz TF chain:
            map -> odom -> base_footprint

        This prevents RViz Fixed Frame 'map' and RobotModel from disappearing.
        """
        try:
            now = self.get_clock().now().to_msg()

            # map -> odom
            t_map_odom = TransformStamped()
            t_map_odom.header.stamp = now
            t_map_odom.header.frame_id = "map"
            t_map_odom.child_frame_id = "odom"
            t_map_odom.transform.translation.x = 0.0
            t_map_odom.transform.translation.y = 0.0
            t_map_odom.transform.translation.z = 0.0
            t_map_odom.transform.rotation.x = 0.0
            t_map_odom.transform.rotation.y = 0.0
            t_map_odom.transform.rotation.z = 0.0
            t_map_odom.transform.rotation.w = 1.0

            transforms = [t_map_odom]

            # odom -> base_footprint from latest Habitat odom
            msg = getattr(self, "latest_odom_msg_for_tf", None)
            if msg is not None:
                t_odom_base = TransformStamped()
                t_odom_base.header.stamp = now
                t_odom_base.header.frame_id = "odom"
                t_odom_base.child_frame_id = "base_footprint"
                t_odom_base.transform.translation.x = float(msg.pose.pose.position.x)
                t_odom_base.transform.translation.y = float(msg.pose.pose.position.y)
                t_odom_base.transform.translation.z = float(msg.pose.pose.position.z)
                t_odom_base.transform.rotation = msg.pose.pose.orientation
                transforms.append(t_odom_base)

            self.tf_broadcaster.sendTransform(transforms)

        except Exception as e:
            if not hasattr(self, "_stable_rviz_tf_warned"):
                print(f"⚠️ Stable RViz TF publish failed: {e}", flush=True)
                self._stable_rviz_tf_warned = True

    def get_odom(self, msg: Odometry):
        self.latest_odom_msg_for_tf = msg

        if not self.setup_complete or self.mfm is None:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        odom_pose = PoseMeters(x, y, yaw)
        self.latest_odom_yaw_for_astar = odom_pose.yaw
        self.robot_yaw = yaw

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
        odom_msg = self.latest_odom_msg_for_tf
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
