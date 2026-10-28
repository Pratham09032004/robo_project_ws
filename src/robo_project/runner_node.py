#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from tf2_ros import TransformBroadcaster

import yaml, os, time, math
from ament_index_python.packages import get_package_share_directory
import numpy as np

from robo_project.scripts.map_handler import MapFrameManager
from robo_project.scripts.basic_types import PoseMeters, PosePixels

# ── Keep-inside-the-room guard ────────────────────────────────────────────────
# Uses the SAME grid, resolution and origin that publish_map() sends on /map
# (and that motion_planner.py plans on), so the guard and the planner always
# agree about what is free space.  Internal convention: 1 = free, 0 = wall.
MAP_RESOLUTION = 0.02       # must match the resolution in publish_map()
FORWARD_LOOKAHEAD_M = 0.10  # don't drive forward if the cell this far ahead is not free

# ── Path follower tuning ──────────────────────────────────────────────────────
WAYPOINT_TOLERANCE_M = 0.25  # intermediate waypoint counts as reached within this distance
WAYPOINT_PASSED_M = 0.40     # ...or within this distance once it is behind the robot
GOAL_TOLERANCE_M = 0.10      # final waypoint (the goal) tolerance
LIN_VEL = 0.3                # m/s when driving
MIN_LIN_VEL = 0.1            # m/s: slow down to this when approaching the goal
MAX_ANG_VEL = 0.8            # rad/s
ANG_GAIN = 1.5               # angular.z = ANG_GAIN * heading error (clamped)
TURN_IN_PLACE_ERR = 0.35     # rad (~20 deg): turn on the spot above this heading error

# ── Stall detection ───────────────────────────────────────────────────────────
# The hand-drawn map is only approximate: Habitat can have an obstacle where
# the sketch shows free space.  While driving forward, compare how far the
# robot was commanded to go with how far it actually moved ALONG ITS HEADING
# (Habitat lets a blocked robot slide sideways along a wall, so plain movement
# is not enough).  If it is far less, report the spot ahead on /blocked_point;
# motion_planner marks it as an obstacle and re-plans from the current pose.
STALL_WINDOW_M = 0.40        # judge after this much commanded forward driving
STALL_MIN_RATIO = 0.30       # stalled if actual forward motion < 30% of commanded
BLOCKED_AHEAD_M = 0.25       # reported obstacle position: this far ahead of the robot


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
        self.blocked_pub = self.create_publisher(PointStamped, '/blocked_point', 10)
        self._reset_stall()

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

        # map -> odom is published by _publish_stable_rviz_tf_timer (20 Hz);
        # not repeated here, so it has a single source.
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        resolution = 0.02
        msg.info.resolution = resolution
        msg.info.height = int(grid_map.shape[0])
        msg.info.width = int(grid_map.shape[1])
        msg.info.origin.position.x = float(-(grid_map.shape[1] / 2.0) * resolution)
        msg.info.origin.position.y = float(-(grid_map.shape[0] / 2.0) * resolution)
        msg.info.origin.orientation.w = 1.0

        # Standard ROS values: 0 = free, 100 = occupied.  OccupancyGrid row 0
        # is the BOTTOM (smallest y) while image row 0 is the top of the
        # sketch, so flip vertically - otherwise RViz shows the map mirrored.
        msg.data = np.where(np.flipud(grid_map) == 1, 0, 100).astype(np.int8).flatten().tolist()
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
        # motion_planner subscribes to /goal_pose too: it re-plans from the
        # current pose and publishes the new /planned_path, which
        # path_callback() then follows.
        print(f"🎯 Goal received from RViz: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
              f"— waiting for motion_planner to re-plan.", flush=True)

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

            # NOTE: odom -> base_footprint is deliberately NOT published here.
            # habitat_bridge already publishes it (same data, from the simulator),
            # and two publishers of the same transform make RViz jitter.
            # runner_node owns ONLY map -> odom.
            self.tf_broadcaster.sendTransform([t_map_odom])

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
        self._reset_stall()
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

        # GUARD 1: robot is outside the free area of the sketch (Habitat's floor
        # can extend past a drawn wall).  Don't stop for good: the planned path
        # starts at the nearest free cell, so keep following it back inside;
        # only the look-ahead check (GUARD 2) is skipped while outside.
        outside = not self._is_free_world(rx, ry)
        if outside:
            self._log_throttled(
                "outside",
                f"⚠️ Robot at ({rx:.2f},{ry:.2f}) is outside the free area of the map — "
                f"driving back along the path.")

        # Skip every waypoint already within tolerance (the planner's path is
        # dense - one point per 2 cm cell - so many are passed per tick).
        while self.waypoint_idx < len(self.path):
            wx, wy = self.path[self.waypoint_idx]
            is_last = self.waypoint_idx == len(self.path) - 1
            d = math.hypot(wx - rx, wy - ry)
            if is_last:
                reached = d < GOAL_TOLERANCE_M
            else:
                # Also count a nearby waypoint that is already behind the robot
                # as passed; otherwise the robot circles around it.
                behind = abs(self._wrap_angle(math.atan2(wy - ry, wx - rx) - ryaw)) > math.pi / 2
                reached = d < WAYPOINT_TOLERANCE_M or (behind and d < WAYPOINT_PASSED_M)
            if not reached:
                break
            self.waypoint_idx += 1
            self._log_throttled("wp", f"Waypoint {self.waypoint_idx}/{len(self.path)} reached.")

        if self.waypoint_idx >= len(self.path):
            self._publish_cmd_vel(0.0, 0.0)
            self.path_goal_reached = True
            print(f"GOAL REACHED! Robot stopped at ({rx:.2f},{ry:.2f}).", flush=True)
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

        # Proportional steering (a fixed +-0.5 rad/s with a tight threshold
        # overshoots and wiggles left/right around the heading).
        ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, ANG_GAIN * angle_err))
        if self._check_stall(rx, ry, ryaw):
            return

        if abs(angle_err) > TURN_IN_PLACE_ERR:
            self._publish_cmd_vel(0.0, ang)
        else:
            # GUARD 2: the cell just ahead is wall/void -> don't step into it.
            ahead_x = rx + FORWARD_LOOKAHEAD_M * math.cos(ryaw)
            ahead_y = ry + FORWARD_LOOKAHEAD_M * math.sin(ryaw)
            if not outside and not self._is_free_world(ahead_x, ahead_y):
                self._publish_cmd_vel(0.0, 0.0)
                self._log_throttled(
                    "blocked",
                    f"🚧 Blocked: cell {FORWARD_LOOKAHEAD_M:.2f} m ahead of "
                    f"({rx:.2f},{ry:.2f}) is not free. Not driving forward.")
                self._report_blocked(rx, ry, ryaw)
                return
            lin = LIN_VEL
            if self.waypoint_idx == len(self.path) - 1:
                lin = max(MIN_LIN_VEL, min(LIN_VEL, dist))  # slow down for the goal
            self._publish_cmd_vel(lin, ang)
            self._drive_prev = (time.time(), rx, ry, ryaw, lin)

    def _reset_stall(self):
        self._drive_prev = None   # (time, x, y, yaw, commanded speed) of the last forward command
        self._stall_expected = 0.0
        self._stall_actual = 0.0

    def _check_stall(self, rx: float, ry: float, ryaw: float) -> bool:
        """True (and obstacle reported) if commanded forward driving did not move
        the robot forward along its heading."""
        now = time.time()
        prev = self._drive_prev
        self._drive_prev = None
        if prev is not None:
            t0, x0, y0, yaw0, v = prev
            self._stall_expected += v * (now - t0)
            self._stall_actual += (rx - x0) * math.cos(yaw0) + (ry - y0) * math.sin(yaw0)
        if self._stall_expected < STALL_WINDOW_M:
            return False
        expected, actual = self._stall_expected, self._stall_actual
        self._reset_stall()
        if actual >= STALL_MIN_RATIO * expected:
            return False
        print(f"⛔ Stalled at ({rx:.2f},{ry:.2f}): drove {actual:.2f} m of {expected:.2f} m commanded — "
              f"something in Habitat blocks the way that is not on the map.", flush=True)
        self._report_blocked(rx, ry, ryaw)
        return True

    def _report_blocked(self, rx: float, ry: float, direction: float):
        """Stop, publish the blocked spot in `direction` from the robot and wait for a new path."""
        self._publish_cmd_vel(0.0, 0.0)
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = rx + BLOCKED_AHEAD_M * math.cos(direction)
        msg.point.y = ry + BLOCKED_AHEAD_M * math.sin(direction)
        self.blocked_pub.publish(msg)
        print(f"   Reported obstacle at ({msg.point.x:.2f},{msg.point.y:.2f}) — asking motion_planner to re-plan.",
              flush=True)
        # Pause following until the re-planned path arrives (path_callback).
        self.path = []
        self._reset_stall()

    # ── Keep-inside-the-room guard ───────────────────────────────────────────

    def _is_free_world(self, x: float, y: float) -> bool:
        """True if world point (x, y) is a free cell on the map.
        Out-of-bounds counts as NOT free (that is the black void outside the room).
        Uses the same origin/resolution as publish_map(), so it matches /map."""
        if self.mfm is None or self.mfm.map is None:
            return True  # map not loaded yet: don't block anything
        grid = self.mfm.map
        H, W = grid.shape
        col = math.floor(x / MAP_RESOLUTION + W / 2.0)
        row = H - 1 - math.floor(y / MAP_RESOLUTION + H / 2.0)  # image rows are flipped vs /map
        if not (0 <= row < H and 0 <= col < W):
            return False
        return grid[row, col] == 1

    def _log_throttled(self, key: str, text: str, period: float = 1.0):
        now = time.time()
        last = getattr(self, "_last_log_" + key, 0.0)
        if now - last > period:
            print(text, flush=True)
            setattr(self, "_last_log_" + key, now)

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

