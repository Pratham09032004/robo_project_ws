#!/usr/bin/env python3

"""
Visualizer class for live visualization of the CMN pipeline.
Migrated from ROS 1 to ROS 2.
"""

import yaml
import cv2
import numpy as np
from math import sin, cos
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from robo_project.scripts.config_loader import load_config

from robo_project.scripts.basic_types import PosePixels
from robo_project.scripts.map_handler import MapFrameManager


class Visualizer:
    """
    Class to handle updating the live viz with any dynamically changing data.
    """
    verbose = False
    # Most recent data for all vars we want to plot.
    occ_map = None                      # Occupancy grid map displayed in background.
    observation = None                  # Most recent observation image.
    observation_region = None           # Area in front of robot used to generate observations.
    veh_pose_true_px: PosePixels = None # Most recent ground-truth vehicle pose.
    veh_pose_estimate: PosePixels = None # Most recent localization estimate.
    particle_set = None                 # All particles in the particle filter (list of PosePixels).
    planned_path = None                 # Full planned path as list of PosePixels.
    goal_cell: PosePixels = None        # Current goal cell in pixels.
    veh_pose_in_obs_region = None       # Dict of veh pose details relative to observation frame (constant once set).
    veh_pose_displ_len = None           # Arrow length for vehicle pose display (constant once set).
    veh_pose_displ_wid = None           # Arrow width for vehicle pose display (constant once set).

    mfm = None  # Reference to MapFrameManager for map configs.

    def __init__(self):
        self.read_params()

    def read_params(self):
        """
        Read configuration params from the yaml.
        """
        config = load_config()
        self.verbose = config["verbose"]

    def set_observation(self, obs_img, obs_rect=None):
        """
        Set a new observation image to display in the viz.
        @param obs_img  - The observation image.
        @param obs_rect - (optional) The bounding region in front of the robot.
        """
        self.observation = obs_img
        self.observation_region = obs_rect

    def set_map_frame_manager(self, mfm: MapFrameManager):
        """
        Set reference to the map frame manager and initialize map-dependent viz settings.
        @param mfm - MapFrameManager instance already initialized with a map.
        """
        self.mfm = mfm
        self.occ_map = mfm.map_with_border.copy()
        self.set_veh_pose_in_obs_region()

        # Arrow display size — small enough to just show the triangle direction indicator.
        self.veh_pose_displ_len = 10 * self.mfm.map_resolution_desired / self.mfm.map_downscale_ratio
        self.veh_pose_displ_wid = 0.01 * self.mfm.map_resolution_desired / self.mfm.map_downscale_ratio

    def set_veh_pose_in_obs_region(self):
        """
        Compute vehicle pose relative to the observation region for display.
        """
        if not self.mfm.initialized:
            return
        col = self.mfm.veh_px_vert_from_bottom_on_obs - 0.5
        row = self.mfm.obs_width_px // 2 + self.mfm.veh_px_horz_from_center_on_obs
        # Robot pose always faces right in the observation subplot.
        d_col = 0.5 * self.mfm.obs_resolution
        d_row = 0.0
        wid = 0.01 / self.mfm.obs_resolution / self.mfm.map_downscale_ratio
        self.veh_pose_in_obs_region = {
            "x": col, "y": row,
            "dx": d_col, "dy": d_row,
            "width": wid
        }

    def get_updated_img(self):
        """
        Update the plot with all the most recent data and return it as a numpy image.
        @return numpy array (BGR image) of the current visualization.
        """
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)

        ######### LEFT SUBPLOT — Map overview ###########
        ax0 = fig.add_subplot(1, 4, (1, 3))
        ax0.imshow(self.occ_map, cmap="gray", vmin=0, vmax=1)

        # Ground-truth vehicle pose.
        if self.veh_pose_true_px is not None:
            ax0.scatter(self.veh_pose_true_px.c, self.veh_pose_true_px.r,
                        color="blue", label="True Vehicle Pose")
            ax0.arrow(self.veh_pose_true_px.c, self.veh_pose_true_px.r,
                      self.veh_pose_displ_len * cos(self.veh_pose_true_px.yaw),
                      -self.veh_pose_displ_len * sin(self.veh_pose_true_px.yaw),
                      color="blue", width=self.veh_pose_displ_wid,
                      head_width=0.01, head_length=0.5)

        # Localization estimate.
        if self.veh_pose_estimate is not None:
            ax0.scatter(self.veh_pose_estimate.c, self.veh_pose_estimate.r,
                        color="green", label="Vehicle Pose Estimate")
            ax0.arrow(self.veh_pose_estimate.c, self.veh_pose_estimate.r,
                      self.veh_pose_displ_len * cos(self.veh_pose_estimate.yaw),
                      -self.veh_pose_displ_len * sin(self.veh_pose_estimate.yaw),
                      color="green", width=self.veh_pose_displ_wid,
                      zorder=3, head_width=0.01, head_length=0.5)

        # Particle filter set.
        if self.particle_set is not None:
            particles_r = [self.particle_set[i].r for i in range(len(self.particle_set))]
            particles_c = [self.particle_set[i].c for i in range(len(self.particle_set))]
            ax0.scatter(particles_c, particles_r, s=10, color="red", zorder=0, label="All Particles")

        # Planned path.
        if self.planned_path is not None:
            path_r = [self.planned_path[i].r for i in range(len(self.planned_path))]
            path_c = [self.planned_path[i].c for i in range(len(self.planned_path))]
            ax0.scatter(path_c, path_r, s=3, color="purple", zorder=1, label="Planned Path")

        # Goal cell.
        if self.goal_cell is not None:
            ax0.scatter(self.goal_cell.c, self.goal_cell.r, color="yellow", label="Goal Cell")

        # Observation bounding box.
        if self.observation_region is not None:
            box = cv2.boxPoints(self.observation_region)
            box_x_coords = [box[i, 0] for i in range(box.shape[0])] + [box[0, 0]]
            box_y_coords = [box[i, 1] for i in range(box.shape[0])] + [box[0, 1]]
            ax0.plot(box_x_coords, box_y_coords, "r-", zorder=2, label="Observed Area")

        ######### RIGHT SUBPLOT — Local observation ###########
        ax1 = fig.add_subplot(1, 4, 4)
        ax1.set_title("GT local occ\n(rel to robot east)")

        if self.observation is not None:
            ax1.imshow(self.observation, cmap="gray", vmin=0, vmax=1)
            if self.veh_pose_in_obs_region is not None:
                ax1.scatter(self.veh_pose_in_obs_region["x"],
                            self.veh_pose_in_obs_region["y"], color="blue")
                ax1.arrow(self.veh_pose_in_obs_region["x"],
                          self.veh_pose_in_obs_region["y"],
                          self.veh_pose_in_obs_region["dx"],
                          self.veh_pose_in_obs_region["dy"],
                          color="blue", width=self.veh_pose_in_obs_region["width"],
                          zorder=2, head_width=0.01, head_length=0.25)

        ax0.legend(loc="upper left", fontsize="x-small")

        # Render to numpy array.
        canvas.draw()
        buf = canvas.buffer_rgba()
        result_img = np.asarray(buf)
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)