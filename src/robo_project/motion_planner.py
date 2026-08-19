#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry

from robo_project.scripts.astar import Astar
from robo_project.scripts.basic_types import PosePixels

# ── GOAL in world coordinates (meters) ───────────────────────────────────────
GOAL_X = 0.0   # meters
GOAL_Y = 0.0   # meters


class MotionPlanner(Node):

    def __init__(self):
        super().__init__('motion_planner')

        path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/planned_path', path_qos)

        # Publish the goal itself, so any other node (e.g. a visualizer)
        # can read the ACTUAL goal being used instead of hardcoding a
        # separate copy of GOAL_X/GOAL_Y that could drift out of sync.
        self.goal_pub = self.create_publisher(PoseStamped, '/motion_planner_goal', path_qos)

        self.grid = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_width = None
        self.map_height = None
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        self.robot_x = None
        self.robot_y = None
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.planned = False
        self.last_path_msg = None
        self.goal_msg = self._build_goal_msg()
        self.goal_pub.publish(self.goal_msg)
        self.create_timer(0.2, self.try_plan)
        self.create_timer(2.0, self.republish_path)

        self.get_logger().info("motion_planner started. Waiting for /map and /odom...")
        self.get_logger().info(f"Goal: x={GOAL_X}m, y={GOAL_Y}m")

    def _build_goal_msg(self):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = GOAL_X
        msg.pose.position.y = GOAL_Y
        msg.pose.orientation.w = 1.0
        return msg

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def map_callback(self, msg: OccupancyGrid):
        if self.grid is not None:
            return  # only need to load the map once
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.grid = np.array(msg.data, dtype=np.int8).reshape(
            self.map_height, self.map_width)
        self.get_logger().info(
            f"Map received: {self.map_width}x{self.map_height} "
            f"res={self.map_resolution}m origin=({self.map_origin_x:.2f},"
            f"{self.map_origin_y:.2f})")

    def odom_callback(self, msg: Odometry):
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)

    # ── Coordinate conversion ────────────────────────────────────────────────
    # Uses the resolution/origin actually published in /map — never
    # hardcoded separately, so it can't drift out of sync.

    def world_to_pixel(self, x: float, y: float):
        col = int((x - self.map_origin_x) / self.map_resolution)
        row = int((y - self.map_origin_y) / self.map_resolution)
        row = max(0, min(self.map_height - 1, row))
        col = max(0, min(self.map_width - 1, col))
        return row, col

    def pixel_to_world(self, row: int, col: int):
        x = self.map_origin_x + col * self.map_resolution
        y = self.map_origin_y + row * self.map_resolution
        return x, y

    # ── Planning ──────────────────────────────────────────────────────────────

    def try_plan(self):
        if self.planned:
            return
        if self.grid is None or self.robot_x is None:
            return
        self.planned = True
        self.run_astar_and_publish()

    def republish_path(self):
        self.goal_msg.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(self.goal_msg)
        if self.last_path_msg is None:
            return
        self.last_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.last_path_msg)

    def run_astar_and_publish(self):
        start_row, start_col = self.world_to_pixel(self.robot_x, self.robot_y)
        goal_row, goal_col = self.world_to_pixel(GOAL_X, GOAL_Y)

        self.get_logger().info(
            f"Start: world=({self.robot_x:.2f},{self.robot_y:.2f}) "
            f"-> pixel=({start_row},{start_col}) "
            f"value={self.grid[start_row, start_col]}")
        self.get_logger().info(
            f"Goal:  world=({GOAL_X:.2f},{GOAL_Y:.2f}) "
            f"-> pixel=({goal_row},{goal_col}) "
            f"value={self.grid[goal_row, goal_col]}")

        # Raw /map data used directly — no transformation.
        # in_collision() checks value==0, and this map already has
        # wall=0, free=100, matching perfectly (see module docstring).
        astar_map = self.grid

        start_row, start_col = self._snap_to_free(astar_map, start_row, start_col)
        goal_row, goal_col = self._snap_to_free(astar_map, goal_row, goal_col)

        planner = Astar()
        planner.map = astar_map
        planner.include_diagonals = True

        start_px = PosePixels(start_row, start_col)
        goal_px = PosePixels(goal_row, goal_col)

        path = planner.run_astar(start_px, goal_px)

        if not path:
            self.get_logger().error(
                "A* found NO PATH. Check GOAL_X/GOAL_Y are in free space.")
            return

        # astar_simple.run_astar returns the path REVERSED (goal->start),
        # with no smoothing (dense, one waypoint per grid cell) — this
        # matches the original ROS1 logic exactly, unchanged.
        waypoints_px = list(reversed(path))  # start -> goal order

        self.get_logger().info(f"Path found: {len(waypoints_px)} waypoints")

        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for wp in waypoints_px:
            x, y = self.pixel_to_world(int(wp.r), int(wp.c))
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.last_path_msg = msg
        self.path_pub.publish(msg)
        self.get_logger().info(
            f"Path published on /planned_path with {len(msg.poses)} waypoints.")

    def _snap_to_free(self, astar_map, r, c, max_radius=100):
        """Snap to nearest cell where value != 0 (free, per this map's convention)."""
        H, W = astar_map.shape
        r = max(0, min(H - 1, r))
        c = max(0, min(W - 1, c))
        if astar_map[r, c] != 0:
            return r, c
        for radius in range(1, max_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) == radius:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and astar_map[nr, nc] != 0:
                            return nr, nc
        return r, c


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
