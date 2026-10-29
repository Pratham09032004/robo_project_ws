#!/usr/bin/env python3
"""ROS 2 node: CMN localization on the coarse (hand-drawn) map.

Inputs
  /map             the coarse map (published by runner_node from the sketch)
  /cmn/local_occ   predicted local occupancy (local_occ_node)
  /odom            ONLY the change in position between updates and the heading
                   are used - the robot's start position is unknown, exactly as
                   in the paper (heading is assumed known, as in the paper).

Outputs
  /cmn/belief      the belief over the coarse map (OccupancyGrid, for RViz)
  /cmn/pose        current estimate + spread (PoseWithCovarianceStamped)
  /cmn/odom        once localized: the robot pose ON THE MAP, at odometry
                   rate (estimate + odometry since the last update).  Remap the
                   planner's and runner's /odom to this topic.
  /cmd_vel         only BEFORE localization converges (explore=true): drive
                   forward while the predicted occupancy ahead is free, turn
                   towards the freer side otherwise.  Silent afterwards, so
                   runner_node owns /cmd_vel from then on.
  /goal_pose       only when the estimate JUMPS after convergence (the belief
                   switched to another place, e.g. the mirror image of a
                   symmetric room): re-sends the planner's current goal
                   (/motion_planner_goal) so motion_planner re-plans from the
                   corrected pose.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from robo_project.cmn_nav.bayes_filter import CoarseMapBayesFilter, yaw_from_quaternion
from robo_project.cmn_nav.local_grid import base_grid_to_egocentric, egocentric_offsets, to_map_aligned


class CmnLocalizationNode(Node):
    def __init__(self):
        super().__init__('cmn_localization')
        p = self.declare_parameter
        p('cell_size', 0.2)          # m - coarse map resolution of the belief
        p('window_radius', 7)        # cells - local window (2R+1)^2 compared with the map
        p('sharpness', 2.0)          # likelihood exponent (1.0 = paper Eq. 2 exactly)
        p('support_dilation', 3)     # cells the belief may extend past the drawn free space
        p('step_m', 0.15)            # m of travel between filter updates (paper's forward step)
        p('converge_conf', 0.75)     # belief mass near the argmax needed to call it localized
        p('converge_updates', 5)     # ... for this many updates in a row
        p('offset_alpha', 0.3)       # smoothing of corrections after convergence
        p('reset_dist', 1.0)         # m - corrections larger than this are applied at once
        p('explore', True)           # drive around until localized
        p('explore_lin', 0.25)
        p('explore_ang', 0.6)
        p('log_gt_error', False)     # EVALUATION ONLY: compare to absolute /odom (valid in Habitat)
        g = lambda n: self.get_parameter(n).value
        self.cfg = {n: g(n) for n in ['cell_size', 'window_radius', 'sharpness', 'support_dilation', 'step_m',
                                      'converge_conf', 'converge_updates', 'offset_alpha', 'reset_dist',
                                      'explore', 'explore_lin', 'explore_ang', 'log_gt_error']}

        self.filter = None
        self.map_header = None
        self.odom = None              # latest Odometry
        self.last_update_xy = None    # odom position at the last filter update
        self.local_occ = None         # egocentric prediction
        self.offset = None            # map pose = odom position + offset
        self.locked = False
        self._streak = 0
        self.n_updates = 0
        self.planner_goal = None      # latest /motion_planner_goal, re-sent on relocalization

        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(OccupancyGrid, '/cmn/local_occ', self.local_occ_cb, 1)
        self.belief_pub = self.create_publisher(OccupancyGrid, '/cmn/belief', latched)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/cmn/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/cmn/odom', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(PoseStamped, '/motion_planner_goal',
                                 lambda m: setattr(self, 'planner_goal', m), latched)
        self.create_timer(0.1, self.explore_tick)
        self.get_logger().info(f"CMN localization: {self.cfg}")

    # ── inputs ────────────────────────────────────────────────────────────────
    def map_cb(self, msg: OccupancyGrid):
        if self.filter is not None:
            return
        grid = np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width)
        c = self.cfg
        self.filter = CoarseMapBayesFilter(
            grid == 0, msg.info.resolution, (msg.info.origin.position.x, msg.info.origin.position.y),
            cell_size=c['cell_size'], window_radius=c['window_radius'], sharpness=c['sharpness'],
            support_dilation=c['support_dilation'])
        self.map_header = msg.header
        h, w = self.filter.shape
        self.get_logger().info(f"Coarse map {w}x{h} cells of {self.filter.cell:.2f} m; "
                               f"belief starts uniform over {int(self.filter.free.sum())} cells")
        self.publish_belief()

    def odom_cb(self, msg: Odometry):
        self.odom = msg
        if self.locked:
            self.publish_map_odom(msg)

    def local_occ_cb(self, msg: OccupancyGrid):
        grid = np.array(msg.data, np.float32).reshape(msg.info.height, msg.info.width) / 100.0
        self.local_occ = base_grid_to_egocentric(grid)
        self.maybe_update()

    # ── filter ────────────────────────────────────────────────────────────────
    def maybe_update(self):
        if self.filter is None or self.odom is None or self.local_occ is None:
            return
        pos = self.odom.pose.pose.position
        xy = np.array([pos.x, pos.y])
        yaw = yaw_from_quaternion(self.odom.pose.pose.orientation)
        if self.last_update_xy is not None:
            d = xy - self.last_update_xy
            if np.hypot(*d) < self.cfg['step_m']:
                return                     # wait until the robot has moved one step
            self.filter.predict(float(d[0]), float(d[1]))
        self.last_update_xy = xy
        self.filter.correct(to_map_aligned(self.local_occ, yaw, self.filter.cell, self.filter.R))
        self.n_updates += 1

        ex, ey, conf, _ = self.filter.estimate()
        target = np.array([ex, ey]) - xy
        if self.offset is None or np.hypot(*(target - self.offset)) > self.cfg['reset_dist']:
            if self.locked and self.offset is not None:
                self.relocalized(target)
            self.offset = target
        else:
            self.offset = self.offset + self.cfg['offset_alpha'] * (target - self.offset)

        self._streak = self._streak + 1 if conf >= self.cfg['converge_conf'] else 0
        if not self.locked and self._streak >= self.cfg['converge_updates']:
            self.locked = True
            self.offset = target
            self.cmd_pub.publish(Twist())
            self.get_logger().info(
                f"LOCALIZED after {self.n_updates} updates: estimate ({ex:.2f}, {ey:.2f}) "
                f"confidence {conf:.2f}. Publishing /cmn/odom - the planner can start.")

        msg = f"update {self.n_updates}: estimate ({ex:.2f},{ey:.2f}) conf {conf:.2f}"
        if self.cfg['log_gt_error']:
            msg += f" | eval: error vs Habitat pose {math.hypot(ex - xy[0], ey - xy[1]):.2f} m"
        self.get_logger().info(msg, throttle_duration_sec=2.0)
        self.publish_pose(ex, ey, yaw)
        self.publish_belief()

    def relocalized(self, target):
        jump = float(np.hypot(*(target - self.offset)))
        self.get_logger().warn(f"Estimate jumped {jump:.2f} m (belief moved to another place) - "
                               f"asking motion_planner to re-plan from the corrected pose.")
        if self.planner_goal is not None:
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose = self.planner_goal.pose
            self.goal_pub.publish(goal)

    # ── outputs ───────────────────────────────────────────────────────────────
    def publish_map_odom(self, odom: Odometry):
        out = Odometry()
        out.header.stamp = odom.header.stamp
        out.header.frame_id = 'map'
        out.child_frame_id = odom.child_frame_id
        out.pose.pose.position.x = odom.pose.pose.position.x + float(self.offset[0])
        out.pose.pose.position.y = odom.pose.pose.position.y + float(self.offset[1])
        out.pose.pose.orientation = odom.pose.pose.orientation
        out.twist = odom.twist
        self.odom_pub.publish(out)

    def publish_pose(self, x, y, yaw):
        b = self.filter.belief
        rr, cc = np.mgrid[0:b.shape[0], 0:b.shape[1]]
        X, Y = self.filter.cell_to_world(rr, cc)
        mx, my = (b * X).sum(), (b * Y).sum()
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        cov = np.zeros(36)
        cov[0] = (b * (X - mx) ** 2).sum()
        cov[7] = (b * (Y - my) ** 2).sum()
        cov[1] = cov[6] = (b * (X - mx) * (Y - my)).sum()
        cov[35] = 0.01
        msg.pose.covariance = cov.tolist()
        self.pose_pub.publish(msg)

    def publish_belief(self):
        f = self.filter
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = f.cell
        msg.info.height, msg.info.width = f.shape
        msg.info.origin.position.x, msg.info.origin.position.y = float(f.origin[0]), float(f.origin[1])
        msg.info.origin.position.z = 0.01
        msg.info.origin.orientation.w = 1.0
        b = f.belief / f.belief.max()
        msg.data = np.round(b * 100).astype(np.int8).flatten().tolist()
        self.belief_pub.publish(msg)

    # ── exploration until localized ─────────────────────────────────────────────
    def explore_tick(self):
        if self.locked or not self.cfg['explore'] or self.local_occ is None:
            return
        f, r = egocentric_offsets()
        F, R = np.meshgrid(f, r, indexing='ij')
        ahead = (F > 0.2) & (F < 0.7) & (np.abs(R) < 0.25)
        left = (F > -0.2) & (F < 1.0) & (R < -0.25) & (R > -1.2)
        right = (F > -0.2) & (F < 1.0) & (R > 0.25) & (R < 1.2)
        occ = self.local_occ
        cmd = Twist()
        if occ[ahead].mean() < 0.35:
            cmd.linear.x = float(self.cfg['explore_lin'])
            # steer gently away from the more occupied side
            cmd.angular.z = 0.5 * float(occ[right].mean() - occ[left].mean())
        else:
            turn_left = occ[left].mean() <= occ[right].mean()
            cmd.angular.z = float(self.cfg['explore_ang']) * (1.0 if turn_left else -1.0)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CmnLocalizationNode()
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
