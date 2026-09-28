import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robo_project'


def generate_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ]

    # Include config files, CMN scripts and the ML model weights recursively in
    # the installed share folder (the nodes look for weights there).
    dirs_to_include = ['config', 'scripts', 'ml_runs']

    for directory in dirs_to_include:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            source_files = [
                os.path.join(root, f)
                for f in files
                if not f.endswith(('.zip', '.pyc', '.pyo'))
            ]
            if source_files:
                dest_dir = os.path.join('share', package_name, root)
                data_files.append((dest_dir, source_files))

    # Pose-estimator weights loaded by ml_pipeline/infer.py.
    pose_weights = glob(os.path.join('ml_pipeline', '*.pth'))
    if pose_weights:
        data_files.append((os.path.join('share', package_name, 'ml_pipeline'), pose_weights))

    return data_files


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name] + [f"{package_name}.{pkg}" for pkg in find_packages(where='.', exclude=['test'])],
    package_dir={package_name: '.'},
    data_files=generate_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='VineBot project team',
    maintainer_email='robot_@todo.todo',
    description='Integrated VineBot simulation, ML, CMN localization, and planning package',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # Existing simulation / planner / ML pose nodes
            'runner_node = robo_project.runner_node:main',
            
            'ml_pose_node = robo_project.ml_pipeline.ml_pose_node:main',
            'pose_monitor_node = robo_project.ml_pipeline.pose_monitor_node:main',
            'bridge = robo_project.habitat_bridge_vinebot_2:main',
            'map_server_node = robo_project.ml_pipeline.map_server_node:main',

            # Jay Week-10 localization validation nodes
            'dataset_gt_replay_node = robo_project.dataset_gt_replay_node:main',
            'ground_truth_local_occ_node = robo_project.dataset_gt_replay_node:main',
            'particle_filter_validation_node = robo_project.particle_filter_validation_node:main',
            'pose_topic_bridge = robo_project.scripts.pose_topic_bridge:main',

            # Jay local-occupancy ML utilities
            'train_local_occupancy = robo_project.ml_local_occ.train_local_occupancy:main',
            'test_local_occupancy_prediction = robo_project.ml_local_occ.test_local_occupancy_prediction:main',

            'motion_planner = robo_project.motion_planner:main',
            'vinebot_interface = robo_project.vinebot_interface:main',
        ],
    },
)
