#!/usr/bin/env python3

"""
Functions ported from Chengguang Xu's original CoarseMapNav class.
Migrated from ROS 1 to ROS 2.
"""

import yaml, os
import numpy as np
import cv2

import rclpy.logging
from robo_project.scripts.config_loader import get_pkg_path, load_config

_logger = rclpy.logging.get_logger('cmn_ported')

from robo_project.scripts.cmn.topo_map import TopoMap, compute_similarity_iou, up_scale_grid, compute_similarity_mse
from robo_project.scripts.cmn.cmn_visualizer import CoarseMapNavVisualizer

from PIL import Image

from robo_project.scripts.cmn.model.local_occupancy_predictor import LocalOccNet
import torch
from torchvision.transforms import Compose, Normalize, PILToTensor

from robo_project.scripts.map_handler import MapFrameManager
from robo_project.scripts.basic_types import yaw_to_cardinal_dir, PosePixels, rotate_image_to_north, cardinal_dir_to_yaw
from robo_project.scripts.astar import Astar


def compute_norm_heuristic_vec(loc_1, loc_2):
    arr_1 = np.array(loc_1)
    arr_2 = np.array(loc_2)
    heu_vec = arr_2 - arr_1
    return heu_vec / np.linalg.norm(heu_vec)


class CoarseMapNavDiscrete:
    """
    Original functions from Chengguang Xu, modified to suit the format/architecture of this project.
    """
    mfm: MapFrameManager = None
    visualizer: CoarseMapNavVisualizer = CoarseMapNavVisualizer()
    astar: Astar = Astar()
    send_random_commands: bool = False
    enable_sim: bool = False
    fuse_lidar_with_rgb: bool = False
    assume_yaw_is_known: bool = True
    ind_to_orientation = ["north", "east", "south", "west"]

    coarse_map_arr = None
    goal_cell: PosePixels = None
    agent_pose_estimate_px: PosePixels = None

    local_occ_net_config = None
    device = None
    model = None
    transformation = None

    coarse_map_graph = None

    predictive_belief_map = None
    observation_prob_map = None
    agent_belief_map = None

    current_local_map = None
    noise_trans_prob = None
    is_facing_a_wall_in_pred_local_occ: bool = False

    def __init__(self, mfm: MapFrameManager, skip_load_model: bool = False,
                 send_random_commands: bool = False, assume_yaw_is_known: bool = True):
        if mfm is None:
            return

        self.mfm = mfm
        self.coarse_map_arr = self.mfm.map_with_border.astype(int)
        self.astar.map = self.mfm.map_with_border.copy()
        self.visualizer.coarse_map = self.mfm.map_with_border.copy()

        self.send_random_commands = send_random_commands

        pkg_share = get_pkg_path()
        cmn_model_dir = os.path.join(pkg_share, "scripts", "cmn", "model")

        config = load_config()
        device_str = config["model"]["device"]
        self.local_occ_net_config = config["model"]["local_occ_net"]

        if not skip_load_model:
            path_to_model = os.path.join(
                cmn_model_dir,
                "trained_local_occupancy_predictor_model.pt"
            )
            self.load_ml_model(path_to_model, device_str)

        self.coarse_map_graph = TopoMap(self.coarse_map_arr, self.mfm.obs_height_px, self.mfm.obs_width_px)

        init_belief = self.coarse_map_arr.copy()
        self.agent_belief_map = init_belief / init_belief.sum()

        self.assume_yaw_is_known = assume_yaw_is_known
        if not self.assume_yaw_is_known:
            self.agent_belief_map = np.repeat(self.agent_belief_map[:, :, np.newaxis], 4, axis=2)

    def set_goal_cell(self, goal_cell: PosePixels):
        self.goal_cell = goal_cell
        if self.coarse_map_arr[self.goal_cell.r, self.goal_cell.c] != 1.0:
            _logger.warn("CMN: Goal cell given in CMN set_goal_cell() is not free!")
        self.astar.goal_cell = goal_cell
        self.visualizer.goal_cell = goal_cell

    def load_ml_model(self, path_to_model: str, device_str: str = "cpu"):
        self.device = torch.device(device_str)
        model = LocalOccNet(self.local_occ_net_config)
        model.load_state_dict(torch.load(path_to_model, map_location="cpu"))
        model.eval()
        self.model = model.to(self.device)

        self.transformation = Compose([
            PILToTensor(),
            lambda x: x.float(),
            lambda x: x / 255.0,
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            lambda x: x.unsqueeze(dim=0),
            lambda x: x.to(self.device)
        ])

    def update_beliefs(self, action: str, agent_yaw: float, facing_a_wall: bool = False):
        if action == "goal_reached":
            return

        if self.predictive_belief_map is None:
            self.predictive_belief_map = self.agent_belief_map.copy()
            self.observation_prob_map = np.zeros_like(self.agent_belief_map)
        else:
            if action == "move_forward":
                if not self.enable_sim:
                    self.noise_trans_prob = np.random.rand()
                else:
                    self.noise_trans_prob = 1

            if action != "move_forward" or not facing_a_wall:
                if self.assume_yaw_is_known:
                    self.predictive_update_func(action, yaw_to_cardinal_dir(agent_yaw))
                    self.agent_pose_estimate_px.apply_action(action)
                else:
                    if action == "move_forward":
                        for i, dir in enumerate(self.ind_to_orientation):
                            self.predictive_update_func(action, dir, i)
                    else:
                        shift = 1 if action == "turn_right" else -1
                        self.predictive_belief_map = np.roll(
                            self.predictive_belief_map, shift=shift, axis=2)

            if self.assume_yaw_is_known:
                measurement_prob_map = self.measurement_update_func()
                self.observation_prob_map = measurement_prob_map / (
                    np.max(measurement_prob_map) + 1e-8)
            else:
                for i in range(4):
                    self.observation_prob_map[:, :, i] = self.measurement_update_func()
                    self.current_local_map = np.rot90(self.current_local_map, k=1)

            rel_weight_pred_vs_obs: float = 0.5

            log_belief = (np.log((1 - rel_weight_pred_vs_obs) * self.observation_prob_map + 1e-8) +
                          np.log(rel_weight_pred_vs_obs * self.predictive_belief_map + 1e-8))
            belief = np.exp(log_belief)
            normalized_belief = belief / belief.sum()
            self.agent_belief_map = normalized_belief.copy()

        self.visualizer.predictive_belief_map = self.predictive_belief_map
        self.visualizer.observation_prob_map = self.observation_prob_map
        self.visualizer.agent_belief_map = self.agent_belief_map

    def predictive_update_func(self, agent_act: str, agent_dir: str, dir_ind: int = None):
        trans_dir_dict = {
            'east':  {'shift':  1, 'axis': 1},
            'west':  {'shift': -1, 'axis': 1},
            'north': {'shift': -1, 'axis': 0},
            'south': {'shift':  1, 'axis': 0}
        }
        shift = trans_dir_dict[agent_dir]['shift']
        axis = trans_dir_dict[agent_dir]['axis']

        if agent_act == "move_forward":
            movable_locations = np.roll(self.coarse_map_arr, shift=-shift, axis=axis)
            movable_locations = np.multiply(self.coarse_map_arr, movable_locations)
            movable_locations = movable_locations + self.coarse_map_arr
            space_to_wall_cells = np.where(movable_locations == 1.0, 1.0, 0.0)

            if self.assume_yaw_is_known:
                current_belief = self.agent_belief_map.copy()
            else:
                current_belief = self.agent_belief_map[:, :, dir_ind].copy()

            noise_trans_move_prob = self.noise_trans_prob
            noise_trans_stay_prob = 1 - self.noise_trans_prob

            pred_stay_belief = np.where(
                space_to_wall_cells == 1.0,
                current_belief,
                current_belief * noise_trans_stay_prob
            )
            pred_move_belief = np.roll(current_belief, shift=shift, axis=axis)
            pred_move_belief = np.multiply(pred_move_belief, self.coarse_map_arr)
            pred_move_belief = pred_move_belief * noise_trans_move_prob
            pred_belief = pred_stay_belief + pred_move_belief
        else:
            if self.assume_yaw_is_known:
                pred_belief = self.agent_belief_map.copy()
            else:
                pred_belief = self.agent_belief_map[dir_ind].copy()

        if self.assume_yaw_is_known:
            self.predictive_belief_map = pred_belief
        else:
            self.predictive_belief_map[:, :, dir_ind] = pred_belief

    def measurement_update_func(self) -> np.ndarray:
        measurement_prob_map = np.zeros_like(self.coarse_map_arr).astype(float)
        for m in self.coarse_map_graph.local_maps:
            candidate_loc = m['loc']
            candidate_map = up_scale_grid(m['map_arr'])
            score = compute_similarity_mse(self.current_local_map, candidate_map)
            measurement_prob_map[candidate_loc[0], candidate_loc[1]] = score
        return measurement_prob_map

    def prediction_to_local_map(self, pred_local_occ: np.ndarray,
                                 agent_yaw: float = None,
                                 lidar_local_occ_meas=None):
        if agent_yaw is not None or not self.assume_yaw_is_known:
            if self.fuse_lidar_with_rgb and lidar_local_occ_meas is not None:
                lidar_occ_facing_NORTH = rotate_image_to_north(lidar_local_occ_meas, 0)
                pred_local_occ = np.mean(np.array([pred_local_occ, lidar_occ_facing_NORTH]), axis=0)

            top_center_cell_block = pred_local_occ[
                :pred_local_occ.shape[0] // 3,
                pred_local_occ.shape[0] // 3:2 * pred_local_occ.shape[0] // 3
            ]
            top_center_cell_mean = np.mean(top_center_cell_block)
            self.is_facing_a_wall_in_pred_local_occ = top_center_cell_mean <= 0.75

            if self.assume_yaw_is_known:
                pred_local_occ = rotate_image_to_north(pred_local_occ, agent_yaw)
                self.visualizer.robot_direction = yaw_to_cardinal_dir(agent_yaw)

        return pred_local_occ

    def predict_local_occupancy(self, pano_rgb, agent_yaw: float = None,
                                 gt_observation=None, lidar_local_occ_meas=None):
        if pano_rgb is not None:
            if self.model is None:
                _logger.error("Cannot predict_local_occupancy() because the model was not loaded!")
                return

            obs = Image.fromarray(pano_rgb)
            pano_rgb_obs_tensor = self.transformation(obs)

            with torch.no_grad():
                pred_local_occ = self.model(pano_rgb_obs_tensor)
                pred_local_occ = pred_local_occ.cpu().squeeze(dim=0).squeeze(dim=0).numpy()

            pred_local_occ = 1 - pred_local_occ
            self.current_local_map = self.prediction_to_local_map(
                pred_local_occ, agent_yaw, lidar_local_occ_meas)
            self.visualizer.pano_rgb = pano_rgb
            self.visualizer.current_predicted_local_map = self.current_local_map

        if gt_observation is not None:
            pred_local_occ = up_scale_grid(gt_observation)
            self.current_local_map = self.prediction_to_local_map(pred_local_occ, agent_yaw)
            self.visualizer.current_ground_truth_local_map = self.current_local_map

    def cmn_localizer(self, agent_yaw: float):
        if self.assume_yaw_is_known:
            candidates = np.where(self.agent_belief_map == self.agent_belief_map.max())
            candidates = [[r, c] for r, c in zip(
                candidates[0].tolist(), candidates[1].tolist())]
            current_est_still_good = (
                self.agent_pose_estimate_px is not None and
                [self.agent_pose_estimate_px.r, self.agent_pose_estimate_px.c] in candidates
            )
        else:
            candidates = np.where(self.agent_belief_map == self.agent_belief_map.max())
            candidates = [[r, c, i] for r, c, i in zip(
                candidates[0].tolist(), candidates[1].tolist(), candidates[2].tolist())]
            current_est_still_good = False
            if self.agent_pose_estimate_px is not None:
                cur_est_yaw_index = self.ind_to_orientation.index(
                    self.agent_pose_estimate_px.get_direction())
                current_est_still_good = (
                    [self.agent_pose_estimate_px.r,
                     self.agent_pose_estimate_px.c,
                     cur_est_yaw_index] in candidates
                )

        if not current_est_still_good:
            rnd_idx = np.random.randint(low=0, high=len(candidates))
            local_map_loc = tuple(candidates[rnd_idx])
            if self.assume_yaw_is_known:
                self.agent_pose_estimate_px = PosePixels(
                    local_map_loc[0], local_map_loc[1], agent_yaw)
            else:
                agent_yaw = cardinal_dir_to_yaw[self.ind_to_orientation[local_map_loc[2]]]
                self.agent_pose_estimate_px = PosePixels(
                    local_map_loc[0], local_map_loc[1], agent_yaw)

        self.visualizer.current_localization_estimate = self.agent_pose_estimate_px

    def choose_next_action(self, agent_yaw: float,
                            true_agent_pose: PosePixels = None) -> str:
        self.cmn_localizer(agent_yaw)

        if (self.agent_pose_estimate_px.r == self.goal_cell.r and
                self.agent_pose_estimate_px.c == self.goal_cell.c):
            _logger.warn("CMN: Estimate matches goal cell, and confidence is {:.3f}".format(
                self.agent_belief_map.max()))
            if self.agent_belief_map.max() > 0.7:
                return "goal_reached"
            else:
                self.agent_belief_map = self.coarse_map_arr / self.coarse_map_arr.sum()

        if self.send_random_commands:
            return np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

        enable_exploration: bool = True
        if enable_exploration:
            if self.agent_belief_map.max() < 0.005:
                _logger.warn(
                    "CMN: Localization has not converged enough, so exploring rather than planning.")
                if self.is_facing_a_wall_in_pred_local_occ:
                    return np.random.choice(['turn_left', 'turn_right'], 1)[0]
                else:
                    return "move_forward"

        if true_agent_pose is not None:
            action = self.astar.get_next_discrete_action(true_agent_pose)
        else:
            action = self.astar.get_next_discrete_action(self.agent_pose_estimate_px)

        self.visualizer.planned_path_to_goal = self.astar.last_path_px_reversed

        if action == "move_forward" and self.is_facing_a_wall_in_pred_local_occ:
            _logger.warn(
                "CMN: A* tried to plan move_forward, but we're facing a wall, so randomly turning.")
            return np.random.choice(['turn_left', 'turn_right'], 1)[0]

        return action

