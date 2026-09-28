#!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
Migrated from ROS 1 to ROS 2.
"""

import yaml
from math import radians, pi, sqrt, remainder, tau
from random import random, randint
from time import sleep  # ROS 2: use time.sleep instead of rospy.sleep
from geometry_msgs.msg import Twist, Vector3

# ROS 2 imports
import rclpy
import rclpy.logging
from robo_project.scripts.config_loader import load_config

from robo_project.scripts.map_handler import clamp, MapFrameManager
from robo_project.scripts.astar import Astar
from robo_project.scripts.pure_pursuit import PurePursuit
from robo_project.scripts.basic_types import PoseMeters, PosePixels, yaw_to_cardinal_dir, cardinal_dir_to_yaw

# Module-level logger for utility classes that aren't nodes themselves
_logger = rclpy.logging.get_logger('motion_controller')


class MotionPlanner:
    """
    Class to send commands to the robot.
    """
    verbose = False
    move_goal_after_reaching: bool = False
    # Publisher set externally by a ROS 2 node.
    cmd_vel_pub = None

    obstacle_in_front_of_robot: bool = False

    # Utility class instances.
    astar = Astar()
    pure_pursuit = PurePursuit()
    mfm = None
    goal_pos_px = None  # PosePixels instance.

    test_motion_types = ["NONE", "RANDOM", "CIRCLE", "STRAIGHT"]
    cur_test_motion_type = None

    path_px_reversed = None  # List of PosePixels objects.

    odom = PoseMeters(0, 0, 0)

    def __init__(self):
        self.read_params()

    def read_params(self):
        """
        Read configuration params from the yaml.
        """
        config = load_config()
        self.verbose = config["verbose"]
        self.astar.verbose = self.verbose
        self.pure_pursuit.verbose = self.verbose
        self.min_lin_vel = config["constraints"]["min_lin_vel"]
        self.max_lin_vel = config["constraints"]["max_lin_vel"]
        self.min_ang_vel = config["constraints"]["min_ang_vel"]
        self.max_ang_vel = config["constraints"]["max_ang_vel"]
        self.do_path_planning = config["path_planning"]["do_path_planning"]
        self.pure_pursuit.use_finite_lookahead_dist = self.do_path_planning
        self.move_goal_after_reaching = config["move_goal_after_reaching"]

    def set_vel_pub(self, pub):
        """
        Set our publisher for velocity commands.
        """
        self.cmd_vel_pub = pub

    def set_odom(self, odom_pose: PoseMeters):
        """
        Update our motion progress based on a new odometry measurement.
        """
        self.odom = odom_pose

    def pub_velocity_cmd(self, fwd, ang):
        """
        Clamp a velocity command within valid values, and publish it to the vehicle.
        """
        fwd = clamp(fwd, 0, self.max_lin_vel)
        ang = clamp(ang, -self.max_ang_vel, self.max_ang_vel)
        if self.verbose:
            # ROS 2: use rclpy logger
            _logger.info("MP: Publishing a command ({:}, {:})".format(fwd, ang))

        if self.cmd_vel_pub is not None:
            # ROS 2: Twist construction — set fields explicitly (no positional args)
            msg = Twist()
            msg.linear.x = float(fwd)
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = float(ang)
            self.cmd_vel_pub.publish(msg)

    def set_test_motion_type(self, type: str):
        """
        Set the type of motion that will be constantly commanded to the robot.
        """
        if type not in self.test_motion_types:
            _logger.warn("MP: Cannot set invalid test motion type {:}. Setting to NONE".format(type))
            self.cur_test_motion_type = "NONE"
        else:
            self.cur_test_motion_type = type

    def cmd_test_motion(self):
        """
        Publish one of the test motions every iteration.
        """
        fwd, ang = 0.0, 0.0
        if self.cur_test_motion_type == "NONE":
            fwd, ang = 0.0, 0.0
        elif self.cur_test_motion_type == "CIRCLE":
            fwd, ang = self.max_lin_vel, self.max_ang_vel
        elif self.cur_test_motion_type == "STRAIGHT":
            fwd, ang = self.max_lin_vel, 0.0
        elif self.cur_test_motion_type == "RANDOM":
            _logger.warn("MP: Test motion type RANDOM is not yet implemented. Sending zero velocity.")
        self.pub_velocity_cmd(fwd, ang)

    def set_map_frame_manager(self, mfm: MapFrameManager):
        """
        Set our reference to the map frame manager.
        """
        self.mfm = mfm
        self.astar.map = self.mfm.map_with_border.copy()

    def set_goal_point_random(self):
        """
        Select a random free cell on the map to use as the goal.
        """
        self.goal_pos_px = self.mfm.choose_random_free_cell()

    def set_goal_point(self, goal_cell: PosePixels):
        """
        Set the goal point in pixels on the map.
        """
        if self.verbose:
            _logger.info("MP: Got goal cell ({:}, {:})".format(int(goal_cell.r), int(goal_cell.c)))
        self.goal_pos_px = goal_cell

    def plan_path_to_goal(self, veh_pose_est: PoseMeters):
        """
        Use A* to generate a path to the current goal cell.
        @return fwd, ang - velocities to command.
        """
        if self.goal_pos_px is None:
            _logger.error("MP: Cannot generate a path to the goal cell, since the goal has not been set. Commanding zero velocity.")
            return 0.0, 0.0

        veh_pose_est_px = self.mfm.transform_pose_m_to_px(veh_pose_est)

        if veh_pose_est_px.distance(self.goal_pos_px) < 2:
            return None, None

        if self.do_path_planning:
            self.path_px_reversed = self.astar.run_astar(veh_pose_est_px, self.goal_pos_px)
            if self.path_px_reversed is None:
                _logger.error("MOT: No path found by A*. Publishing zeros for motion command.")
                return 0.0, 0.0
        else:
            self.path_px_reversed = [self.goal_pos_px, veh_pose_est_px]

        if self.verbose:
            _logger.info("MOT: Planned path from A*: " + ",".join([str(pose) for pose in self.path_px_reversed]))

        path = []
        for i in range(len(self.path_px_reversed) - 1, -1, -1):
            path.append(self.mfm.transform_pose_px_to_m(self.path_px_reversed[i]))
            if self.mfm.map_with_border[self.path_px_reversed[i].r, self.path_px_reversed[i].c] == 0:
                if self.verbose:
                    _logger.warn("MOT: Path contains an occluded cell.")

        fwd, ang = self.pure_pursuit.compute_command(veh_pose_est, path)

        fwd_clamped = clamp(fwd, 0, self.max_lin_vel)
        ang_clamped = clamp(ang, -self.max_ang_vel, self.max_ang_vel)
        if self.verbose and (fwd != fwd_clamped or ang != ang_clamped):
            _logger.warn("MOT: Clamped pure pursuit output from ({:.2f}, {:.2f}) to ({:.2f}, {:.2f}).".format(
                fwd, ang, fwd_clamped, ang_clamped))

        return fwd_clamped, ang_clamped


class MotionTracker:
    """
    Utility to keep track of how far the robot has pivoted since being initialized.
    No ROS dependencies — unchanged from ROS 1.
    """
    last_yaw = None
    ang_motion = 0.0

    def __init__(self):
        pass

    def reset(self):
        self.last_yaw = None
        self.ang_motion = 0.0

    def update_for_pivot(self, new_yaw):
        dtheta = 0.0
        if self.last_yaw is not None:
            dtheta = new_yaw - self.last_yaw
            if abs(dtheta) > pi:
                dtheta = 2 * pi - abs(new_yaw) - abs(self.last_yaw)
                if self.last_yaw < 0 and new_yaw > 0:
                    dtheta = -dtheta
        self.last_yaw = new_yaw
        self.ang_motion += dtheta
        return self.ang_motion


class PController:
    """
    P-only controller for basic motion control.
    No ROS dependencies — unchanged from ROS 1.
    """
    last_v: float = None
    p = 0.1

    def __init__(self, init_value: float = 0.0, p: float = 0.1):
        self.last_v = init_value
        self.p = p

    def update(self, target_v: float):
        self.last_v = target_v * self.p + self.last_v * (1 - self.p)
        return self.last_v


class DiscreteMotionPlanner(MotionPlanner):
    """
    Class to command discrete actions to the robot.
    """
    discrete_actions = ["turn_left", "turn_right", "move_forward"]
    motion_tracker = MotionTracker()
    wait_for_motion_to_complete: bool = True
    command_pivots_globally: bool = True
    lin_goal_reach_deviation: float = None
    ang_goal_reach_deviation: float = None

    def __init__(self):
        self.read_params()

    def read_params(self):
        super().read_params()
        config = load_config()
        self.discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
        self.lin_goal_reach_deviation = abs(config["goal_reach_deviation"]["linear"])
        self.ang_goal_reach_deviation = radians(abs(config["goal_reach_deviation"]["angular"]))

    def cmd_discrete_action(self, action: str):
        """
        Command a discrete action.
        @return fwd, ang distances moved.
        """
        fwd = self.discrete_forward_dist if action == "move_forward" else 0.0
        ang = radians(-90.0 if action == "turn_right" else (90.0 if action == "turn_left" else 0.0))

        if self.wait_for_motion_to_complete:
            _logger.info("DMP: Commanding discrete action {:}.".format(action))
            if action in ["turn_left", "turn_right"]:
                if self.command_pivots_globally:
                    self.cmd_discrete_ang_motion_global(ang)
                else:
                    self.cmd_discrete_ang_motion_relative(ang)
            elif action == "move_forward":
                self.cmd_discrete_fwd_motion(fwd)
            else:
                _logger.warn("DMP: Invalid discrete action {:} cannot be commanded.".format(action))
                fwd, ang = 0.0, 0.0

            self.pub_velocity_cmd(0, 0)
            # ROS 2: use time.sleep instead of rospy.sleep
            sleep(0.5)

        return fwd, ang

    def cmd_random_discrete_action(self):
        """
        Choose a random action and command it.
        @return fwd, ang distances moved.
        """
        return self.cmd_discrete_action(self.discrete_actions[randint(0, len(self.discrete_actions) - 1)])

    def cmd_discrete_ang_motion_global(self, angle: float):
        """
        Turn the robot in-place by a discrete amount, snapping to the nearest cardinal direction.
        """
        final_dir: str = yaw_to_cardinal_dir(self.odom.yaw + angle)
        self.cmd_pivot_to_face_direction(final_dir)

    def cmd_pivot_to_face_direction(self, final_dir: str):
        """
        Command a pivot to align with the specified cardinal direction.
        """
        final_yaw: float = cardinal_dir_to_yaw[final_dir]
        actual_amount_to_turn = remainder(final_yaw - self.odom.yaw, tau)

        if self.verbose:
            _logger.info("DMP: Commanding a discrete pivot from {:} to {:}. starting yaw: {:.3f}, goal yaw: {:.3f}".format(
                self.odom.get_direction(), final_dir, self.odom.yaw, final_yaw))

        turn_dir_sign = actual_amount_to_turn / abs(actual_amount_to_turn)
        remaining_turn_rads = abs(actual_amount_to_turn)

        while remaining_turn_rads > self.ang_goal_reach_deviation:
            if actual_amount_to_turn > 0.5:
                abs_ang_vel_to_cmd = remaining_turn_rads / abs(actual_amount_to_turn) * self.max_ang_vel
            else:
                abs_ang_vel_to_cmd = 0

            abs_ang_vel_to_cmd = max(abs_ang_vel_to_cmd, self.min_ang_vel)
            self.pub_velocity_cmd(0, abs_ang_vel_to_cmd * turn_dir_sign)
            # ROS 2: use time.sleep instead of rospy.sleep
            sleep(0.001)

            rads_to_go = remainder(final_yaw - self.odom.yaw, tau)
            remaining_turn_rads = abs(rads_to_go)
            if rads_to_go * turn_dir_sign < 0:
                break

    def cmd_discrete_ang_motion_relative(self, angle: float):
        """
        Turn the robot in-place by a discrete relative amount.
        """
        if self.verbose:
            _logger.info("DMP: Commanding a discrete pivot of {:} radians.".format(angle))

        turn_dir_sign = angle / abs(angle)
        self.motion_tracker.reset()
        remaining_turn_rads = abs(angle)

        while remaining_turn_rads > self.ang_goal_reach_deviation:
            if angle > 0.5:
                abs_ang_vel_to_cmd = remaining_turn_rads / abs(angle) * self.max_ang_vel
            else:
                abs_ang_vel_to_cmd = 0

            abs_ang_vel_to_cmd = max(abs_ang_vel_to_cmd, self.min_ang_vel)
            self.pub_velocity_cmd(0, abs_ang_vel_to_cmd * turn_dir_sign)
            # ROS 2: use time.sleep instead of rospy.sleep
            sleep(0.001)
            remaining_turn_rads = abs(angle) - abs(self.motion_tracker.update_for_pivot(self.odom.yaw))

    def cmd_discrete_fwd_motion(self, dist: float):
        """
        Move the robot forwards by a discrete distance, then stop.
        """
        if self.verbose:
            _logger.info("DMP: Commanding a discrete forward motion of {:} meters.".format(dist))

        motion_sign = dist / abs(dist)
        init_odom = self.odom
        pid = PController(0.0, 0.0001)
        ramp_threshold = 0.5 * dist
        remaining_motion = dist - sqrt((self.odom.x - init_odom.x) ** 2 + (self.odom.y - init_odom.y) ** 2)

        while remaining_motion > self.lin_goal_reach_deviation and not self.obstacle_in_front_of_robot:
            if remaining_motion > ramp_threshold:
                v = pid.update(self.max_lin_vel)
            else:
                v = pid.update(self.min_lin_vel)

            self.pub_velocity_cmd(v * motion_sign, -0.008)
            # ROS 2: use time.sleep instead of rospy.sleep
            sleep(0.001)
            remaining_motion = dist - sqrt((self.odom.x - init_odom.x) ** 2 + (self.odom.y - init_odom.y) ** 2)

        self.pub_velocity_cmd(0.0, 0.0)

        if self.obstacle_in_front_of_robot:
            _logger.warn("DMP: Stopping forward motion due to obstacle.")