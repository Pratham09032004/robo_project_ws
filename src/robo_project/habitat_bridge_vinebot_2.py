#!/usr/bin/env python3

"""
Bridge between Habitat-sim and ROS 2 for the VineBot.

Publishes:  /camera/{front,right,back,left}/image_raw, /odom, TF odom -> base_footprint
Subscribes: /cmd_vel (continuous velocities, integrated every tick on the navmesh)

Frame convention (matches resources/habitat_map_metadata.json and the datasets):
    ros_x = habitat_x, ros_y = habitat_z
This frame is mirrored relative to Habitat, so a ROS yaw psi corresponds to a
Habitat rotation of theta = -pi/2 - psi about +Y (Habitat agents face -Z).

Run from the workspace (needs the conda env with habitat_sim):
    python3 src/robo_project/habitat_bridge_vinebot_2.py
    python3 src/robo_project/habitat_bridge_vinebot_2.py --ros-args -p scene_path:=/path/to/scene.glb
"""

import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from cv_bridge import CvBridge
import habitat_sim
import habitat_sim.utils.common
import numpy as np
import cv2

# --- Default scene: <workspace>/src/environment/IMR_lab8.glb ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_CANDIDATES = [
    SCRIPT_DIR.parent / "environment" / "IMR_lab8.glb",                      # run from src/robo_project
    Path.home() / "robo_project_ws" / "src" / "environment" / "IMR_lab8.glb",  # installed via ros2 run
]
DEFAULT_SCENE = next((str(p) for p in DEFAULT_SCENE_CANDIDATES if p.exists()),
                     str(DEFAULT_SCENE_CANDIDATES[0]))

UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def habitat_heading_to_ros_yaw(rotation):
    """
    Convert a Habitat agent rotation into a yaw in the ROS map frame.

    Positions are published as ros_x = habitat_x, ros_y = habitat_z (see
    resources/habitat_map_metadata.json). Habitat agents face -Z and turn
    about +Y, so copying the quaternion's y/w into a ROS z/w quaternion gives
    a heading that is off by -90 deg and mirrored. Instead, rotate the
    agent's forward vector (0, 0, -1) and measure its angle in the same
    (x, z) plane the position uses.
    """
    try:
        w = float(rotation.scalar)
        x, y, z = (float(v) for v in rotation.vector)
    except AttributeError:
        w, x, y, z = float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)

    # v' = v + 2w(u x v) + 2u x (u x v), with u = (x, y, z), v = (0, 0, -1)
    cx, cy, cz = -y, x, 0.0
    ccx, ccz = y * cz - z * cy, x * cy - y * cx
    fwd_x = 2.0 * (w * cx + ccx)
    fwd_z = -1.0 + 2.0 * (w * cz + ccz)
    return float(np.arctan2(fwd_z, fwd_x))


def ros_yaw_to_habitat_rotation(yaw):
    """Inverse of habitat_heading_to_ros_yaw() for an upright agent."""
    theta = -math.pi / 2.0 - yaw
    return habitat_sim.utils.common.quat_from_angle_axis(theta, UP_AXIS)


class VinebotHabitatBridge(Node):
    def __init__(self):
        super().__init__('habitat_bridge')
        self.cv_bridge = CvBridge()

        self.declare_parameter('scene_path', DEFAULT_SCENE)
        self.declare_parameter('max_lin_vel', 1.0)    # m/s, clamp on /cmd_vel
        self.declare_parameter('max_ang_vel', 2.0)    # rad/s, clamp on /cmd_vel
        self.declare_parameter('cmd_timeout', 0.5)    # s without /cmd_vel -> stop
        self.declare_parameter('show_window', True)   # OpenCV window with the 4 cameras
        # Optional start pose in the map frame. By default the robot starts at a random
        # point on the FLOOR (Habitat's own random start can land on the roof or a table).
        self.declare_parameter('start_x', float('nan'))
        self.declare_parameter('start_y', float('nan'))
        self.declare_parameter('start_yaw', float('nan'))
        self.scene_path = os.path.expanduser(str(self.get_parameter('scene_path').value))
        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.show_window = bool(self.get_parameter('show_window').value)

        self.get_logger().info(f"📂 Loading Habitat 3D Scene Model from: {self.scene_path}")

        # Camera Publishers
        self.pub_front = self.create_publisher(Image, '/camera/front/image_raw', 10)
        self.pub_right = self.create_publisher(Image, '/camera/right/image_raw', 10)
        self.pub_back  = self.create_publisher(Image, '/camera/back/image_raw', 10)
        self.pub_left  = self.create_publisher(Image, '/camera/left/image_raw', 10)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # TF Broadcaster to move the robot in RViz
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # File Existence Check
        if not os.path.exists(self.scene_path):
            self.get_logger().error(
                f"❌ CRITICAL ERROR: Could not find {self.scene_path}! "
                f"Pass --ros-args -p scene_path:=/path/to/scene.glb")
            raise FileNotFoundError(f"Missing 3D mesh at {self.scene_path}")

        self.sim = self.setup_habitat()
        self.agent = self.sim.initialize_agent(0)
        self.place_agent_on_floor()

        # Latest velocity command and when it arrived.
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0
        self.cmd_time = 0.0
        self.last_tick = time.monotonic()
        self._stuck_count = 0

        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("✅ Habitat Bridge fully initialized.")

    def setup_habitat(self):
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path

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

    def find_floor_height(self, samples=400):
        """
        Height of the floor level of the navmesh. Random navigable points are
        area-weighted and include the roof and table tops, so take the lowest
        height band that holds a real share of the samples.
        """
        pts = [np.asarray(self.sim.pathfinder.get_random_navigable_point(), dtype=np.float32) for _ in range(samples)]
        pts = [p for p in pts if np.all(np.isfinite(p))]
        heights = np.array([p[1] for p in pts])
        for h in np.sort(np.unique(np.round(heights, 1))):
            band = [p for p in pts if abs(p[1] - h) < 0.25]
            if len(band) >= 0.05 * len(pts):
                return float(h), band
        return float(heights.min()), pts

    def place_agent_on_floor(self):
        floor_y, floor_pts = self.find_floor_height()
        state = self.agent.get_state()
        sx = float(self.get_parameter('start_x').value)
        sy = float(self.get_parameter('start_y').value)
        syaw = float(self.get_parameter('start_yaw').value)
        if math.isfinite(sx) and math.isfinite(sy):
            target = np.array([sx, floor_y, sy], dtype=np.float32)   # map (x, y) = habitat (x, z)
            pos = np.asarray(self.sim.pathfinder.snap_point(target), dtype=np.float32)
            if not np.all(np.isfinite(pos)) or abs(pos[1] - floor_y) > 0.3:
                self.get_logger().warn(f"start_x/start_y ({sx:.2f}, {sy:.2f}) is not on the floor; using a random floor point.")
                pos = floor_pts[np.random.randint(len(floor_pts))]
        else:
            pos = floor_pts[np.random.randint(len(floor_pts))]
        state.position = np.asarray(pos, dtype=np.float32)
        if math.isfinite(syaw):
            state.rotation = ros_yaw_to_habitat_rotation(syaw)
        self.agent.set_state(state)
        self.get_logger().info(
            f"Robot starts on the floor (height {floor_y:.2f} m) at map x={pos[0]:.2f}, y={pos[2]:.2f}, "
            f"yaw={math.degrees(habitat_heading_to_ros_yaw(state.rotation)):.0f} deg")

    def cmd_vel_callback(self, msg):
        # Only store the command; it is integrated over the real elapsed time
        # in apply_velocity(), so the robot moves at the commanded speed
        # regardless of how often /cmd_vel is published.
        self.cmd_lin = float(np.clip(msg.linear.x, -self.max_lin_vel, self.max_lin_vel))
        self.cmd_ang = float(np.clip(msg.angular.z, -self.max_ang_vel, self.max_ang_vel))
        self.cmd_time = time.monotonic()

    def apply_velocity(self, dt):
        """Integrate the latest /cmd_vel for dt seconds, constrained to the navmesh."""
        if time.monotonic() - self.cmd_time > self.cmd_timeout:
            return  # no recent command: stand still
        lin, ang = self.cmd_lin, self.cmd_ang
        if abs(lin) < 1e-4 and abs(ang) < 1e-4:
            return

        state = self.agent.get_state()
        yaw = habitat_heading_to_ros_yaw(state.rotation) + ang * dt
        # Move along the heading at the middle of the step (exact for arcs to first order).
        mid_yaw = yaw - 0.5 * ang * dt
        step = lin * dt
        start = np.asarray(state.position, dtype=np.float32)
        target = start + np.array([math.cos(mid_yaw) * step, 0.0, math.sin(mid_yaw) * step], dtype=np.float32)
        # try_step keeps the agent on the navmesh and slides it along walls.
        new_pos = np.asarray(self.sim.pathfinder.try_step(start, target), dtype=np.float32)

        moved = float(np.linalg.norm((new_pos - start)[[0, 2]]))
        if abs(step) > 1e-3 and moved < 0.1 * abs(step):
            self._stuck_count += 1
            if self._stuck_count == 1 or self._stuck_count % 20 == 0:
                self.get_logger().warn(
                    f"Motion blocked by the navmesh boundary (stuck_count={self._stuck_count}). "
                    f"Robot holds its position; the planner/controller must turn away.")
        else:
            self._stuck_count = 0

        state.position = new_pos
        state.rotation = ros_yaw_to_habitat_rotation(yaw)
        self.agent.set_state(state)

    def add_label(self, img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (160, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def timer_callback(self):
        now = time.monotonic()
        dt = min(now - self.last_tick, 0.5)
        self.last_tick = now
        self.apply_velocity(dt)

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
            cv2.imshow('Habitat Live Cameras', grid)

            # Manual driving in the camera window (discrete Habitat actions).
            key = cv2.waitKey(1) & 0xFF
            if key == ord("w"): self.agent.act("move_forward")
            elif key == ord("s"): self.agent.act("move_backward")
            elif key == ord("a"): self.agent.act("turn_left")
            elif key == ord("d"): self.agent.act("turn_right")

        # Pose & Transform Extraction
        state = self.agent.get_state()

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"

        odom_msg.pose.pose.position.x = float(state.position[0])
        odom_msg.pose.pose.position.y = float(state.position[2])
        odom_msg.pose.pose.position.z = 0.0

        # Heading expressed in the same frame as the position above.
        yaw = habitat_heading_to_ros_yaw(state.rotation)
        odom_msg.pose.pose.orientation.z = float(np.sin(yaw / 2.0))
        odom_msg.pose.pose.orientation.w = float(np.cos(yaw / 2.0))

        if now - self.cmd_time <= self.cmd_timeout:
            odom_msg.twist.twist.linear.x = self.cmd_lin
            odom_msg.twist.twist.angular.z = self.cmd_ang

        self.odom_pub.publish(odom_msg)

        # Broadcast the exact same pose to the TF Tree for RViz
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = 0.0
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    bridge = VinebotHabitatBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
