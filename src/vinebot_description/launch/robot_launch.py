import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_path = os.path.join(get_package_share_directory('vinebot_description'))
    xacro_file = os.path.join(pkg_path, 'urdf', 'vinebot.xacro')
    robot_description_config = xacro.process_file(xacro_file).toxml()

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_joint_state_publisher = LaunchConfiguration('use_joint_state_publisher')

    params = {'robot_description': robot_description_config, 'use_sim_time': use_sim_time}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the /clock from Gazebo (true when launched from launch_sim.launch.py)'),
        DeclareLaunchArgument(
            'use_joint_state_publisher', default_value='true',
            description='Publish /joint_states (disable in Gazebo, where joint_broad publishes them)'),

        # 1. State Publisher (The Chassis)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[params]
        ),
        # 2. Joint State Publisher (The Wheels/Joints) - Habitat / RViz only
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_joint_state_publisher)
        )
    ])
