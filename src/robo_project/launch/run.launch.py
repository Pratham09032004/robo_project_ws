#!/usr/bin/env python3

"""
ROS 2 launch file for the Habitat VineBot navigation stack.

Starts:
  - map_server_node : the only publisher of /map and the static map -> odom TF
  - motion_planner  : A* from /odom to the goal, publishes /planned_path
                      (replans whenever a "2D Goal Pose" is clicked in RViz)
  - runner_node     : follows /planned_path, publishes /cmd_vel

The Habitat bridge (habitat_bridge_vinebot_2.py) is started separately, since
it needs the conda environment with habitat_sim.

Usage:
  ros2 launch robo_project run.launch.py goal_x:=1.0 goal_y:=0.5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    goal_x_arg = DeclareLaunchArgument(
        'goal_x', default_value='0.0', description='Initial goal x in the map frame (m)')
    goal_y_arg = DeclareLaunchArgument(
        'goal_y', default_value='0.0', description='Initial goal y in the map frame (m)')

    map_server_node = Node(
        package='robo_project',
        executable='map_server_node',
        name='static_map_server_node',
        output='screen',
        emulate_tty=True,
    )

    motion_planner_node = Node(
        package='robo_project',
        executable='motion_planner',
        name='motion_planner',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'goal_x': ParameterValue(LaunchConfiguration('goal_x'), value_type=float),
            'goal_y': ParameterValue(LaunchConfiguration('goal_y'), value_type=float),
        }],
    )

    runner_node = Node(
        package='robo_project',
        executable='runner_node',
        name='runner_node',
        output='screen',
        emulate_tty=True,   # Ensures coloured log output in the terminal.
    )

    return LaunchDescription([
        goal_x_arg,
        goal_y_arg,
        map_server_node,
        motion_planner_node,
        runner_node,
    ])
