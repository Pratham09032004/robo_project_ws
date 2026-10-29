"""CMN stack: ML local occupancy + localization on the sketch + the existing
A* planner / path follower, which now get the LOCALIZED pose instead of the
simulator's ground truth (their /odom is remapped to /cmn/odom).

Start habitat_bridge_vinebot_2.py (and the robot model / RViz) first, then:
    ros2 launch robo_project cmn.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_gt = LaunchConfiguration('log_gt_error')
    model_path = LaunchConfiguration('model_path')
    return LaunchDescription([
        DeclareLaunchArgument('log_gt_error', default_value='false',
                              description='Evaluation only: log error vs the Habitat ground-truth pose'),
        DeclareLaunchArgument('model_path', default_value='',
                              description='Local occupancy model (default: ml_runs/cmn_local_occ.pth)'),
        Node(package='robo_project', executable='runner_node', output='screen',
             remappings=[('/odom', '/cmn/odom')]),
        Node(package='robo_project', executable='local_occ_node', output='screen',
             parameters=[{'model_path': model_path}]),
        Node(package='robo_project', executable='cmn_localization_node', output='screen',
             parameters=[{'log_gt_error': log_gt}]),
        Node(package='robo_project', executable='motion_planner', output='screen',
             remappings=[('/odom', '/cmn/odom')]),
    ])
