#!/usr/bin/env python3

"""
Single place that decides which config yaml the project uses.

By default this is <share>/robo_project/config/config.yaml. Set the
ROBO_PROJECT_CONFIG environment variable to use another one, either a file
name inside the config folder (e.g. config_delton.yaml) or an absolute path.
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory


def get_pkg_path() -> str:
    return get_package_share_directory('robo_project')


def get_config_path() -> str:
    name = os.environ.get('ROBO_PROJECT_CONFIG', 'config.yaml')
    name = os.path.expanduser(name)
    if os.path.isabs(name):
        return name
    return os.path.join(get_pkg_path(), 'config', name)


def load_config() -> dict:
    with open(get_config_path(), 'r') as file:
        return yaml.safe_load(file)
