import os

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():


    package_name='vinebot_description'
    world_package_name='vineyard_world'


    default_world = os.path.join(
        get_package_share_directory(world_package_name),
        'world',
        'real_vineyard_for_result.sdf'
    )

    world = LaunchConfiguration('world')

    # Let Gazebo resolve model://vineyard_world/... (world assets) and
    # package://vinebot_description/... (robot meshes) from the installed share folders.
    resource_paths = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.pathsep.join([
            os.path.dirname(get_package_share_directory(world_package_name)),
            os.path.dirname(get_package_share_directory(package_name)),
        ]))

    world_args= DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='sdf to load'
        )

    gz_args_extra = DeclareLaunchArgument(
        'gz_args_extra',
        default_value='',
        description="Extra gz sim arguments, e.g. '-s --headless-rendering' to run without the GUI"
        )
    

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("vinebot_description"),
            "config",
            "my_controller.yaml",
        ]
    )
    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','robot_launch.py'
                )]), launch_arguments={'use_sim_time': 'true', 'use_joint_state_publisher': 'false'}.items()
    )

    
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
                    launch_arguments={'gz_args': ['-r -v4 --render-engine=ogre2 ', LaunchConfiguration('gz_args_extra'), ' ', world], 'on_exit_shutdown': 'true'}.items()
             )

    
    spawn_entity = Node(package='ros_gz_sim', executable='create',
                        arguments=['-topic', 'robot_description',
                                   '-name', 'vinebot',
                                   '-y','5.0',
                                   '-z','10.0'],
                        output='screen')
    
    
    
    # One spawner for both controllers: they are activated together, and generous
    # timeouts keep it working when the simulation runs slower than real time.
    controllers_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_cont",
            "joint_broad",
            "--param-file",
            robot_controllers,
            "--controller-manager-timeout", "60",
            "--switch-timeout", "60",
            "--service-call-timeout", "60",
        ],
        output="screen"
    )

    
    
    bridge_config = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'gz_bridge.yaml'
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_config}',
        ],
        output='screen'
    )

    return LaunchDescription([

        resource_paths,
        rsp,
        world_args,
        gz_args_extra,
        gazebo,
        spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[controllers_spawner]
            )
        ),

        # /clock, lidar (/scan, /points) and /imu from Gazebo.
        ros_gz_bridge,
            
    ])
