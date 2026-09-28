#!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
Migrated from ROS 1 to ROS 2.
"""

import yaml, cv2, os
import numpy as np
from math import sin, cos, remainder, tau, ceil, pi
from random import random, randrange
from cv_bridge import CvBridge, CvBridgeError

# ROS 2 imports
import rclpy
import rclpy.logging

from robo_project.scripts.config_loader import get_pkg_path, load_config
from robo_project.scripts.rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle
from robo_project.scripts.basic_types import PoseMeters, PosePixels, Pose

#### GLOBAL VARIABLES ####
bridge = CvBridge()
#########################

_logger = rclpy.logging.get_logger('robo_project')

def clamp(val: float, min_val: float, max_val: float):
    return min(max(min_val, val), max_val)

class CoarseMapProcessor:
    pkg_path = None  
    verbose = False
    show_map_images = False
    map_fpath = None
    obs_balloon_radius = 0
    map_resolution_raw = None
    map_resolution_desired = None
    map_downscale_ratio = None
    raw_map = None
    occ_map = None
    inv_occ_map = None

    def __init__(self):
        self.pkg_path = get_pkg_path()

        config = load_config()
        self.verbose = config["verbose"]
        self.show_map_images = config["map"]["show_images_during_pre_proc"]
        self.map_fpath = os.path.join(self.pkg_path, "config/maps", config["map"]["fname"])
        self.obs_balloon_radius = config["map"]["obstacle_balloon_radius"]

        map_name = os.path.splitext(config["map"]["fname"])[0]
        map_yaml_fpath = os.path.join(self.pkg_path, "config/maps", map_name + ".yaml")
        if not os.path.exists(map_yaml_fpath):
            map_yaml_fpath = os.path.join(self.pkg_path, "config/maps/default.yaml")
        with open(map_yaml_fpath, 'r') as file2:
            map_config = yaml.safe_load(file2)
            self.map_resolution_raw = map_config["resolution"]
            self.map_occ_thresh_min = map_config.get("occ_thresh_min", 0.1)
            self.map_occ_thresh_max = map_config.get("occ_thresh_max", 0.9)
            # "sketch" maps are photos of hand-drawn maps (white paper = free, pen = wall).
            self.map_type = map_config.get("type", "image")
            self.sketch_scale_percent = map_config.get("scale_percent", 100)
            self.sketch_threshold = map_config.get("threshold", 150)

        self.map_resolution_desired = config["map"]["desired_meters_per_pixel"]
        self.map_downscale_ratio = self.map_resolution_raw / self.map_resolution_desired

        if not os.path.exists(self.map_fpath):
            raise FileNotFoundError(f"Map file from config not found: {self.map_fpath}")
        self.read_coarse_map_from_file()
        _logger.info(f"Loaded map {self.map_fpath} ({self.occ_map.shape[1]}x{self.occ_map.shape[0]} px)")

    def read_sketch_map(self):
        """
        Load a photo of a hand-drawn map. The photo is shrunk so it doesn't
        exhaust memory, then thresholded: white paper (> threshold) = 1 (free),
        pen strokes = 0 (walls).
        """
        img = cv2.imread(self.map_fpath, cv2.IMREAD_GRAYSCALE)
        width = int(img.shape[1] * self.sketch_scale_percent / 100 * self.map_downscale_ratio)
        height = int(img.shape[0] * self.sketch_scale_percent / 100 * self.map_downscale_ratio)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        _, binary_map = cv2.threshold(img, self.sketch_threshold, 1, cv2.THRESH_BINARY)
        return binary_map

    def read_coarse_map_from_file(self):
        if self.map_type == "sketch":
            occ_map_img = self.read_sketch_map()
            img = occ_map_img
        elif os.path.splitext(self.map_fpath)[1] == ".npy":
            img = np.load(self.map_fpath)
        else:
            img = cv2.imread(self.map_fpath, cv2.IMREAD_UNCHANGED)
            if len(img.shape) >= 3 and img.shape[2] == 4:
                a1 = ~img[:, :, 3]
                img = cv2.add(cv2.merge([a1, a1, a1, a1]), img)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        if self.map_type != "sketch":
            img = cv2.resize(img, (int(img.shape[1] * self.map_downscale_ratio), int(img.shape[0] * self.map_downscale_ratio)), 0, 0, cv2.INTER_AREA)

            if len(img.shape) >= 3 and img.shape[2] >= 3:
                self.raw_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.map_occ_thresh_min, self.map_occ_thresh_max, cv2.THRESH_BINARY)[1]
                occ_map_img = np.divide(occ_map_img, 255)
            else:
                occ_map_img = img

        self.occ_map = np.round(occ_map_img)

        if self.obs_balloon_radius != 0:
            if len(img.shape) >= 3 and img.shape[2] >= 3:
                if self.obs_balloon_radius == 0:
                    self.obs_balloon_radius = 1
            nbrs = []
            for i in range(-self.obs_balloon_radius, self.obs_balloon_radius + 1):
                for j in range(-self.obs_balloon_radius, self.obs_balloon_radius + 1):
                    nbrs.append((i, j))
            nbrs.remove((0, 0))
            for i in range(len(self.occ_map)):
                for j in range(len(self.occ_map[0])):
                    if occ_map_img[i][j] != 1:
                        for chg in nbrs:
                            self.occ_map[clamp(i + chg[0], 0, self.occ_map.shape[0] - 1)][clamp(j + chg[1], 0, self.occ_map.shape[1] - 1)] = 0

        self.occ_map = np.float32(np.array(self.occ_map))
        self.inv_occ_map = np.logical_not(self.occ_map).astype(int)


class MapFrameManager(CoarseMapProcessor):
    initialized = False
    map_with_border = None
    inv_map_with_border = None
    use_discrete_state_space = False
    show_obs_gen_debug: bool = False

    def __init__(self, use_discrete_state_space: bool):
        super().__init__()
        self.use_discrete_state_space = use_discrete_state_space
        config = load_config()
        self.obs_resolution = config["observation"]["resolution"] / self.map_downscale_ratio
        self.obs_height_px = config["observation"]["height"]
        self.obs_width_px = config["observation"]["width"]
        self.obs_height_px_on_map = int(self.obs_height_px * self.obs_resolution / self.map_resolution_desired)
        self.obs_width_px_on_map = int(self.obs_width_px * self.obs_resolution / self.map_resolution_desired)
        self.veh_px_horz_from_center_on_obs = (config["observation"]["veh_horz_pos_ratio"] - 0.5) * self.obs_width_px
        self.veh_px_vert_from_bottom_on_obs = config["observation"]["veh_vert_pos_ratio"] * self.obs_width_px
        self.veh_px_horz_from_center_on_map = self.veh_px_horz_from_center_on_obs * self.obs_resolution / self.map_resolution_desired
        self.veh_px_vert_from_bottom_on_map = self.veh_px_vert_from_bottom_on_obs * self.obs_resolution / self.map_resolution_desired
        self.setup_map()

    def setup_map(self):
        # The map (image, .npy or hand-drawn sketch) comes from config.yaml -> map.fname.
        self.map = np.round(self.occ_map).astype(int)
        max_obs_dim = 3
        self.map_with_border = cv2.copyMakeBorder(self.map, max_obs_dim, max_obs_dim, max_obs_dim, max_obs_dim, cv2.BORDER_CONSTANT, None, 0)
        # transform_map_px_to_m(row, col): bottom-left cell gives the min corner, top-right the max.
        self.map_x_min_meters, self.map_y_min_meters = self.transform_map_px_to_m(self.map_with_border.shape[0] - 1, 0)
        self.map_x_max_meters, self.map_y_max_meters = self.transform_map_px_to_m(0, self.map_with_border.shape[1] - 1)
        self.initialized = True
        self.inv_map_with_border = np.logical_not(self.map_with_border).astype(int)

    def transform_pose_px_to_m(self, pose_px: PosePixels) -> PoseMeters:
        if pose_px is None: return None
        x, y = self.transform_map_px_to_m(pose_px.r, pose_px.c)
        return PoseMeters(x, y, pose_px.yaw)

    def transform_map_px_to_m(self, row: int, col: int):
        row = int(clamp(row, 0, self.map_with_border.shape[0] - 1))
        col = int(clamp(col, 0, self.map_with_border.shape[1] - 1))
        row_offset = row - self.map_with_border.shape[0] // 2
        col_offset = col - self.map_with_border.shape[1] // 2
        x = self.map_resolution_desired * col_offset
        y = self.map_resolution_desired * -row_offset
        return x, y

    def transform_pose_m_to_px(self, pose_m: PoseMeters) -> PosePixels:
        if pose_m is None: return None
        r, c = self.transform_map_m_to_px(pose_m.x, pose_m.y)
        return PosePixels(r, c, pose_m.yaw)

    def transform_map_m_to_px(self, x: float, y: float):
        col_offset = x / self.map_resolution_desired
        row_offset = -y / self.map_resolution_desired
        row = int(row_offset + self.map_with_border.shape[0] // 2)
        col = int(col_offset + self.map_with_border.shape[1] // 2)
        row = int(clamp(row, 0, self.map_with_border.shape[0] - 1))
        col = int(clamp(col, 0, self.map_with_border.shape[1] - 1))
        return row, col

    def extract_observation_region(self, veh_pose: Pose, pose_in_meters: bool = True):
        if pose_in_meters:
            veh_pose_px = self.transform_pose_m_to_px(veh_pose)
        else:
            veh_pose_px = veh_pose
        center_col = veh_pose_px.c + (self.obs_height_px_on_map / 2 - self.veh_px_vert_from_bottom_on_map) * cos(veh_pose_px.yaw)
        center_row = veh_pose_px.r - (self.obs_height_px_on_map / 2 - self.veh_px_vert_from_bottom_on_map) * sin(veh_pose_px.yaw)
        center = (center_col, center_row)
        angle = -np.rad2deg(veh_pose_px.yaw)
        rect = None
        if not self.use_discrete_state_space:
            rect = (center, (self.obs_height_px_on_map, self.obs_width_px_on_map), angle)
            obs_img = crop_rotated_rectangle(image=self.map_with_border, rect=rect)
            if obs_img is None:
                return None, None
        else:
            half_obs_dim = self.obs_height_px_on_map // 2
            obs_img = self.map_with_border[int(center_row) - half_obs_dim:int(center_row) + half_obs_dim + 1, int(center_col) - half_obs_dim:int(center_col) + half_obs_dim + 1]
            agent_dir_str = veh_pose_px.get_direction()
            if agent_dir_str == "east": pass
            elif agent_dir_str == "north": obs_img = np.rot90(obs_img, k=-1)
            elif agent_dir_str == "west": obs_img = np.rot90(obs_img, k=2)
            elif agent_dir_str == "south": obs_img = np.rot90(obs_img, k=1)
            else: raise Exception("Invalid agent direction")
        obs_img = np.asarray(obs_img)
        obs_img = np.nan_to_num(obs_img, nan=0.0, posinf=1.0, neginf=0.0)
        obs_img = np.ascontiguousarray(obs_img, dtype=np.float32)
        obs_img = cv2.resize(
            obs_img,
            (self.obs_height_px, self.obs_width_px),
            interpolation=cv2.INTER_NEAREST,
        )
        return obs_img, rect

    def choose_random_free_cell(self) -> PosePixels:
        while True:
            r = randrange(0, self.map_with_border.shape[0])
            c = randrange(0, self.map_with_border.shape[1])
            if self.map_with_border[r, c] == 1:
                return PosePixels(r, c)

    def generate_random_valid_veh_pose(self, in_meters: bool = True) -> Pose:
        if self.use_discrete_state_space:
            angles = [0.0, pi / 2, pi, -pi / 2]
            yaw = angles[randrange(0, len(angles))]
        else:
            yaw = remainder(random() * tau, tau)
        free_cell: PosePixels = self.choose_random_free_cell()
        free_cell.yaw = yaw
        if in_meters: return self.transform_pose_px_to_m(free_cell)
        else: return free_cell

    def veh_pose_m_in_collision(self, veh_pose_m: PoseMeters) -> bool:
        r, c = self.transform_map_m_to_px(veh_pose_m.x, veh_pose_m.y)
        return self.map_with_border[r, c] != 1

    def veh_pose_px_in_collision(self, veh_pose_px: PosePixels) -> bool:
        return self.map_with_border[veh_pose_px.r, veh_pose_px.c] != 1

class Simulator(MapFrameManager):
    veh_pose_true_px: PosePixels = None
    veh_pose_true_meters: PoseMeters = None
    veh_pose_true_se2 = None
    discrete_forward_dist: float = None

    def __init__(self, use_discrete_state_space):
        super().__init__(use_discrete_state_space)
        config = load_config()
        self.dt = config["dt"]
        self.max_lin_vel = config["constraints"]["max_lin_vel"]
        self.min_ang_vel = config["constraints"]["min_ang_vel"]
        self.max_ang_vel = config["constraints"]["max_ang_vel"]
        self.allow_motion_through_occupied_cells = config["simulator"]["allow_motion_through_occupied_cells"]
        self.discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
        self.show_obs_gen_debug = config["simulator"]["show_obs_gen_debug"]
        self.veh_pose_true_px = self.generate_random_valid_veh_pose(False)
        self.veh_pose_true_meters = self.transform_pose_px_to_m(self.veh_pose_true_px)

    def propagate_with_vel(self, lin: float, ang: float):
        fwd_dist = self.dt * clamp(lin, 0, self.max_lin_vel)
        dtheta = self.dt * clamp(ang, -self.max_ang_vel, self.max_ang_vel)
        self.propagate_with_dist(fwd_dist, dtheta)

    def get_veh_pose_after_motion(self, lin: float, ang: float) -> PoseMeters:
        veh_pose_proposed = PoseMeters()
        veh_pose_proposed.x = self.veh_pose_true_meters.x + lin * cos(self.veh_pose_true_meters.yaw)
        veh_pose_proposed.y = self.veh_pose_true_meters.y + lin * sin(self.veh_pose_true_meters.yaw)
        veh_pose_proposed.x = clamp(veh_pose_proposed.x, self.map_x_min_meters, self.map_x_max_meters)
        veh_pose_proposed.y = clamp(veh_pose_proposed.y, self.map_y_min_meters, self.map_y_max_meters)
        veh_pose_proposed.yaw = remainder(self.veh_pose_true_meters.yaw + ang, tau)
        return veh_pose_proposed

    def propagate_with_dist(self, lin: float, ang: float):
        veh_pose_proposed = self.get_veh_pose_after_motion(lin, ang)
        if not self.allow_motion_through_occupied_cells and self.veh_pose_m_in_collision(veh_pose_proposed):
            self.veh_pose_true_meters.yaw = veh_pose_proposed.yaw
        else:
            self.veh_pose_true_meters = veh_pose_proposed
        self.veh_pose_true_px = self.transform_pose_m_to_px(self.veh_pose_true_meters)

    def propagate_with_discrete_motion(self, action: str):
        if action == "turn_left":
            self.veh_pose_true_px.yaw = remainder(self.veh_pose_true_px.yaw + pi / 2, tau)
        elif action == "turn_right":
            self.veh_pose_true_px.yaw = remainder(self.veh_pose_true_px.yaw - pi / 2, tau)
        elif action == "move_forward":
            agent_dir = self.veh_pose_true_px.get_direction()
            dr = {"east": 0, "west": 0, "north": -1, "south": 1}
            dc = {"east": 1, "west": -1, "north": 0, "south": 0}
            proposed_new_pose = PosePixels(self.veh_pose_true_px.r + dr[agent_dir], self.veh_pose_true_px.c + dc[agent_dir], self.veh_pose_true_px.yaw)
            if not self.veh_pose_px_in_collision(proposed_new_pose):
                self.veh_pose_true_px = proposed_new_pose
        self.veh_pose_true_meters = self.transform_pose_px_to_m(self.veh_pose_true_px)

    def get_true_observation(self):
        return self.extract_observation_region(self.veh_pose_true_px, False)

    def agent_is_facing_wall(self) -> bool:
        return self.veh_pose_m_in_collision(self.get_veh_pose_after_motion(self.discrete_forward_dist, 0.0))
