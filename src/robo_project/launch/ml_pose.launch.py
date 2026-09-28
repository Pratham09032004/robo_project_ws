from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    image_folder_arg = DeclareLaunchArgument(
        "image_folder",
        default_value=[EnvironmentVariable("HOME"), "/robo_project_ws/src/robo_project/habitat_dataset/front"],
        description="Folder with the front-camera images to run the pose estimator on",
    )

    return LaunchDescription([

        image_folder_arg,

        Node(
            package="robo_project",
            executable="ml_pose_node",
            name="ml_pose_node",
            output="screen",

            parameters=[
                {
                    "image_folder": LaunchConfiguration("image_folder")
                }
            ]
        )

    ])
