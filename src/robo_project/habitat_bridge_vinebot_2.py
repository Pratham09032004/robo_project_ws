#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from cv_bridge import CvBridge
import habitat_sim
import numpy as np
import cv2
import os
from pathlib import Path

# --- STRICT PATH SETUP FOR ENVIRONMENT FOLDER ---
SCRIPT_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = SCRIPT_DIR.parent.parent / "src" / "environment" 
MODEL_PATH = str(ENVIRONMENT_DIR / "IMR_lab8.glb")

class VinebotHabitatBridge(Node):
    def __init__(self):
        super().__init__('habitat_bridge')
        self.cv_bridge = CvBridge()
        
        self.get_logger().info(f"📂 Loading Habitat 3D Scene Model from: {MODEL_PATH}")

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
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"❌ CRITICAL ERROR: Could not find {MODEL_PATH}!")
            raise FileNotFoundError(f"Missing 3D mesh at {MODEL_PATH}")

        self.sim = self.setup_habitat()
        self.agent = self.sim.initialize_agent(0)

        self._stuck_count = 0

        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("✅ Habitat Bridge fully initialized with IMR Lab 8.")

    def setup_habitat(self):
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = MODEL_PATH
        
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

    def cmd_vel_callback(self, msg):
        # FIX: previous if/elif chain meant that whenever linear.x > 0.01,
        # angular.z was NEVER checked that same tick even if also nonzero.
        # runner_node.py sends combined linear+angular for course
        # correction while approaching a waypoint — that steering signal
        # was being silently dropped every time the robot moved forward.
        # Now angular is applied independently so steering-while-driving
        # actually works.
        if abs(msg.angular.z) > 0.01:
            if msg.angular.z > 0:
                self.agent.act("turn_left")
            else:
                self.agent.act("turn_right")

        if msg.linear.x > 0.01:
            pos_before = self.agent.get_state().position.copy()
            self.agent.act("move_forward")
            pos_after = self.agent.get_state().position
            moved = float(np.linalg.norm(pos_after - pos_before))

            if moved < 0.01:
                self._stuck_count += 1
                if self._stuck_count == 1 or self._stuck_count % 20 == 0:
                    self.get_logger().warn(
                        f"move_forward rejected near navmesh boundary "
                        f"(stuck_count={self._stuck_count})")

                # FIX: repeated turn_left/turn_right nudging was CONFIRMED
                # ineffective by logs — stuck_count climbed past 2000 over
                # 7+ minutes with ZERO successful recoveries (alternating
                # turns cancel each other's net rotation out to ~zero).
                # Instead, after a short grace period, directly snap the
                # agent to the nearest genuinely navigable point using
                # Habitat's own pathfinder — the same tool that reliably
                # detects navigability elsewhere in this project. This is
                # a hard, guaranteed-correct recovery instead of an
                # unreliable incremental nudge.
                if self._stuck_count >= 10:
                    self._snap_attempts = getattr(self, '_snap_attempts', 0) + 1
                    state = self.agent.get_state()

                    # FIX: repeated snap-to-NEAREST-point kept returning
                    # the SAME coordinates every time (confirmed by logs),
                    # because the agent stays trapped in a tiny isolated
                    # navmesh pocket where snap_point() always finds this
                    # same pocket as "nearest", regardless of random
                    # rotation — the agent never successfully moves out.
                    # After 3 failed snap-to-nearest attempts in a row,
                    # escalate to a genuinely RANDOM navigable point
                    # anywhere in the scene, which should land in a much
                    # larger, more open area instead of the same dead end.
                    if self._snap_attempts >= 3:
                        random_point = self.sim.pathfinder.get_random_navigable_point()
                        state.position = random_point
                        self.get_logger().warn(
                            f"Nearest-point recovery failed {self._snap_attempts} times — "
                            f"escalating to random navigable point: {random_point}")
                        self._snap_attempts = 0
                    else:
                        snapped = self.sim.pathfinder.snap_point(state.position)
                        if snapped is not None:
                            state.position = snapped
                        self.get_logger().warn(
                            f"Stuck for {self._stuck_count} attempts — "
                            f"snapped agent to nearest navigable point: {state.position}")

                    random_yaw_deg = float(np.random.uniform(0, 360))
                    state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
                        np.deg2rad(random_yaw_deg),
                        np.array([0.0, 1.0, 0.0], dtype=np.float32))
                    self.agent.set_state(state)
                    self._stuck_count = 0
            else:
                self._stuck_count = 0
        elif msg.linear.x < -0.01:
            self.agent.act("move_backward")

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
        
        if "front" in obs:
            f = cv2.cvtColor(obs["front"][:, :, :3], cv2.COLOR_RGB2BGR)
            r = cv2.cvtColor(obs["right"][:, :, :3], cv2.COLOR_RGB2BGR)
            b = cv2.cvtColor(obs["back"][:, :, :3], cv2.COLOR_RGB2BGR)
            l = cv2.cvtColor(obs["left"][:, :, :3], cv2.COLOR_RGB2BGR)
            
            top = np.hstack([self.add_label(f, "Front"), self.add_label(l, "Left")])
            bot = np.hstack([self.add_label(r, "Right"), self.add_label(b, "Back")])
            grid = np.vstack([top, bot])
            cv2.imshow('Habitat IMR_Lab8 Live Cameras', grid)
            
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
        
        # Extract true rotation
        try:
            odom_msg.pose.pose.orientation.z = float(state.rotation.vector.y)
            odom_msg.pose.pose.orientation.w = float(state.rotation.scalar)
        except AttributeError:
            odom_msg.pose.pose.orientation.z = float(state.rotation.y)
            odom_msg.pose.pose.orientation.w = float(state.rotation.w)
            
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
