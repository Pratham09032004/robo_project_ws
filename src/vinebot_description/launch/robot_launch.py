import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = os.path.join(get_package_share_directory('vinebot_description'))
    xacro_file = os.path.join(pkg_path, 'urdf', 'vinebot.xacro')
    robot_description_config = xacro.process_file(xacro_file).toxml()
    params = {'robot_description': robot_description_config}

    return LaunchDescription([
        # 1. State Publisher (The Chassis)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[params]
        ),
        # 2. Joint State Publisher (The Wheels/Joints)
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            output='screen',
            parameters=[params]
        )
    ])
