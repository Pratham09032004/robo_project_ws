#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from cv_bridge import CvBridge
import habitat_sim
import numpy as np
import cv2
import math
import os
import threading
import time
from pathlib import Path

# --- STRICT PATH SETUP FOR ENVIRONMENT FOLDER ---
SCRIPT_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = SCRIPT_DIR.parent.parent / "src" / "environment"
MODEL_PATH = str(ENVIRONMENT_DIR / "IMR_lab8.glb")

# ── Frame convention ─────────────────────────────────────────────────────────
# Habitat is Y-up: the floor is the X/Z plane and the agent looks along -Z
# when its yaw (rotation about +Y) is 0.  ROS (REP-103) is Z-up and
# right-handed, so the ROS odom frame used here is:
#
#     ros_x   =  habitat_X
#     ros_y   = -habitat_Z
#     ros_yaw =  habitat_yaw + pi/2
#
# With this, "forward" in Habitat is exactly the direction of ros_yaw, and a
# positive angular.z (turn left) increases ros_yaw - which is what
# runner_node's path follower assumes.  (Publishing y = +Z and yaw = habitat
# yaw made the reported heading 90-180 deg off the real driving direction.)
YAW_OFFSET = math.pi / 2

# Default spawn, in the ROS map/odom frame.  A point inside the main room that
# is free on the sketch map AND on the Habitat navmesh (>0.5 m from obstacles).
DEFAULT_START_X = -4.1
DEFAULT_START_Y = 4.9
DEFAULT_START_YAW = 0.0

CMD_VEL_TIMEOUT = 0.5   # s - stop if no /cmd_vel arrives for this long
MAX_DT = 0.2            # s - cap one integration step (e.g. after a slow render)


def habitat_to_ros(position, habitat_yaw):
    return float(position[0]), float(-position[2]), wrap_angle(habitat_yaw + YAW_OFFSET)


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def habitat_yaw_of(rotation):
    """Yaw about +Y of a Habitat agent rotation (numpy-quaternion)."""
    return 2.0 * math.atan2(float(rotation.y), float(rotation.w))


class VinebotHabitatBridge(Node):
    def __init__(self):
        super().__init__('habitat_bridge')
        self.cv_bridge = CvBridge()

        self.declare_parameter('start_x', DEFAULT_START_X)
        self.declare_parameter('start_y', DEFAULT_START_Y)
        self.declare_parameter('start_yaw', DEFAULT_START_YAW)
        # -1 = no CUDA device (software / Mesa rendering, e.g. a VM or headless box)
        self.declare_parameter('gpu_device_id', 0)
        # Show the 4-camera OpenCV window (also gives WASD teleop)
        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter('show_window').value

        self.get_logger().info(f"📂 Loading Habitat 3D Scene Model from: {MODEL_PATH}")

        # Camera Publishers
        self.pub_front = self.create_publisher(Image, '/camera/front/image_raw', 10)
        self.pub_right = self.create_publisher(Image, '/camera/right/image_raw', 10)
        self.pub_back  = self.create_publisher(Image, '/camera/back/image_raw', 10)
        self.pub_left  = self.create_publisher(Image, '/camera/left/image_raw', 10)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # TF Broadcaster to move the robot in RViz
        self.tf_broadcaster = TransformBroadcaster(self)
        # Queue depth 1: only the newest velocity command matters.
        # ROS callbacks are spun on a background thread (see main()), so a new
        # command is stored immediately; the simulator itself runs in run_loop()
        # on the main thread (its OpenGL context belongs to that thread).  With
        # ROS timers driving the simulator, the always-overdue timers were served
        # before /cmd_vel and commands reached the robot ~1 s late, which made
        # the path follower overshoot.
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 1)

        # File Existence Check
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"❌ CRITICAL ERROR: Could not find {MODEL_PATH}!")
            raise FileNotFoundError(f"Missing 3D mesh at {MODEL_PATH}")

        self.sim = self.setup_habitat()
        self.agent = self.sim.initialize_agent(0)
        self.place_agent_at_start()

        # Latest velocity command; integrated continuously in motion_callback.
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0
        self.last_cmd_time = 0.0
        self.last_motion_time = time.monotonic()
        self._blocked_since = None

        self.get_logger().info("✅ Habitat Bridge fully initialized with IMR Lab 8.")

    MOTION_PERIOD = 0.02   # s - motion + /odom + TF
    CAMERA_PERIOD = 0.1    # s - camera images (+ preview window)

    def run_loop(self):
        """Simulator loop on the main thread: motion every MOTION_PERIOD, cameras
        every CAMERA_PERIOD."""
        next_camera = time.monotonic()
        while rclpy.ok():
            start = time.monotonic()
            self.motion_callback()
            if start >= next_camera:
                self.timer_callback()
                next_camera = start + self.CAMERA_PERIOD
            time.sleep(max(0.0, self.MOTION_PERIOD - (time.monotonic() - start)))

    def setup_habitat(self):
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = MODEL_PATH
        backend_cfg.gpu_device_id = int(self.get_parameter('gpu_device_id').value)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 0.6

        sensors = []
        for name, orientation in [("front", 0.0), ("right", -1.5708), ("back", 3.14159), ("left", 1.5708)]:
            cam = habitat_sim.CameraSensorSpec()
            cam.uuid = name
            cam.sensor_type = habitat_sim.SensorType.COLOR
            cam.resolution = [480, 640]
            cam.position = [0.0, 0.6, 0.0]
            cam.orientation = [0.0, orientation, 0.0]
            sensors.append(cam)
        agent_cfg.sensor_specifications = sensors

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        return habitat_sim.Simulator(cfg)

    # Navmesh heights above this are NOT the floor (desks ~0.8-1.0 m, wall tops
    # / roof ~3.0-3.2 m in IMR_lab8.glb; the floor is at ~0.2 m).
    FLOOR_MAX_Y = 0.5

    def _random_floor_point(self, current_position):
        """Random navigable point at FLOOR height. Falls back to the nearest
        navigable point to the current position if none is found."""
        pf = self.sim.pathfinder
        for _ in range(500):
            p = pf.get_random_navigable_point()
            if float(p[1]) <= self.FLOOR_MAX_Y:
                return p
        self.get_logger().warn(
            "No floor-level random point found in 500 tries — using snap_point instead.")
        snapped = pf.snap_point(current_position)
        return snapped if snapped is not None else current_position

    def place_agent_at_start(self):
        """Put the agent at (start_x, start_y, start_yaw) given in the ROS frame,
        snapped onto the floor navmesh (habitat's default spawn is random and
        landed in a pocket of the map that is walled off on the sketch)."""
        x = float(self.get_parameter('start_x').value)
        y = float(self.get_parameter('start_y').value)
        yaw = float(self.get_parameter('start_yaw').value)

        state = self.agent.get_state()
        wanted = np.array([x, float(state.position[1]), -y], dtype=np.float32)
        snapped = self.sim.pathfinder.snap_point(wanted)
        if snapped is None or np.isnan(snapped).any() or float(snapped[1]) > self.FLOOR_MAX_Y:
            self.get_logger().warn(f"Start ({x:.2f},{y:.2f}) is not on the floor navmesh — using a random floor point.")
            snapped = self._random_floor_point(state.position)
        state.position = snapped
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            yaw - YAW_OFFSET, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.agent.set_state(state)
        rx, ry, ryaw = habitat_to_ros(snapped, yaw - YAW_OFFSET)
        self.get_logger().info(f"🤖 Robot spawned at x={rx:.2f} y={ry:.2f} yaw={math.degrees(ryaw):.0f}°")

    def cmd_vel_callback(self, msg):
        # Velocities are integrated over time in motion_callback (continuous
        # motion), instead of one fixed 0.25 m / 10 deg Habitat action per
        # message, which made the robot jump past waypoints and oscillate.
        self.cmd_lin = float(msg.linear.x)
        self.cmd_ang = float(msg.angular.z)
        self.last_cmd_time = time.monotonic()

    def motion_callback(self):
        now = time.monotonic()
        dt = min(now - self.last_motion_time, MAX_DT)
        self.last_motion_time = now

        if now - self.last_cmd_time > CMD_VEL_TIMEOUT:
            self.cmd_lin = 0.0
            self.cmd_ang = 0.0

        state = self.agent.get_state()
        habitat_yaw = habitat_yaw_of(state.rotation)

        if self.cmd_lin != 0.0 or self.cmd_ang != 0.0:
            habitat_yaw += self.cmd_ang * dt
            start = np.array(state.position, dtype=np.float32)
            fwd = self.cmd_lin * dt
            # Habitat forward for yaw t is (-sin t, 0, -cos t)
            target = start + np.array([-math.sin(habitat_yaw) * fwd, 0.0, -math.cos(habitat_yaw) * fwd],
                                      dtype=np.float32)
            # try_step keeps the agent on the navmesh (slides along walls).
            new_pos = self.sim.pathfinder.try_step(start, target)
            state.position = new_pos
            state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
                habitat_yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            self.agent.set_state(state)

            moved = float(np.linalg.norm(np.asarray(new_pos) - start))
            if abs(fwd) > 1e-4 and moved < 0.2 * abs(fwd):
                if self._blocked_since is None:
                    self._blocked_since = now
                elif now - self._blocked_since > 2.0:
                    self.get_logger().warn("🚧 Driving forward but blocked by the navmesh (wall/obstacle in Habitat).")
                    self._blocked_since = now
            else:
                self._blocked_since = None

        self.publish_odom(state.position, habitat_yaw)

    def publish_odom(self, position, habitat_yaw):
        x, y, yaw = habitat_to_ros(position, habitat_yaw)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        odom_msg.twist.twist.linear.x = self.cmd_lin
        odom_msg.twist.twist.angular.z = self.cmd_ang
        self.odom_pub.publish(odom_msg)

        # Broadcast the exact same pose to the TF Tree for RViz
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def add_label(self, img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (160, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def timer_callback(self):
        obs = self.sim.get_sensor_observations()

        for name in ["front", "right", "back", "left"]:
            if name in obs:
                pub = getattr(self, f"pub_{name}")
                # Convert Habitat RGB matrix to standard ROS 2 BGR image message
                bgr_img = cv2.cvtColor(obs[name][:, :, :3], cv2.COLOR_RGB2BGR)
                pub.publish(self.cv_bridge.cv2_to_imgmsg(bgr_img, "bgr8"))

        if self.show_window and "front" in obs:
            f = cv2.cvtColor(obs["front"][:, :, :3], cv2.COLOR_RGB2BGR)
            r = cv2.cvtColor(obs["right"][:, :, :3], cv2.COLOR_RGB2BGR)
            b = cv2.cvtColor(obs["back"][:, :, :3], cv2.COLOR_RGB2BGR)
            l = cv2.cvtColor(obs["left"][:, :, :3], cv2.COLOR_RGB2BGR)

            top = np.hstack([self.add_label(f, "Front"), self.add_label(l, "Left")])
            bot = np.hstack([self.add_label(r, "Right"), self.add_label(b, "Back")])
            grid = np.vstack([top, bot])
            cv2.imshow('Habitat IMR_Lab8 Live Cameras', grid)

            # WASD teleop: same as a short /cmd_vel burst
            key = cv2.waitKey(1) & 0xFF
            teleop = {ord("w"): (0.5, 0.0), ord("s"): (-0.5, 0.0),
                      ord("a"): (0.0, 1.0), ord("d"): (0.0, -1.0)}
            if key in teleop:
                self.cmd_lin, self.cmd_ang = teleop[key]
                self.last_cmd_time = time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    bridge = VinebotHabitatBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    threading.Thread(target=executor.spin, daemon=True).start()   # /cmd_vel callbacks
    try:
        bridge.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

