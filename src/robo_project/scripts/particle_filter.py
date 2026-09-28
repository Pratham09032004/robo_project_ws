#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates,
and can be polled for the most likely particle estimate at any time.
Migrated from ROS 1 to ROS 2.
"""

import yaml
import numpy as np
from math import sin, cos, remainder, tau
from random import choices

from robo_project.scripts.config_loader import load_config

from robo_project.scripts.map_handler import MapFrameManager
from robo_project.scripts.basic_types import PoseMeters, PosePixels


class ParticleFilter:
    # Config params.
    num_particles = None
    state_size = None
    num_to_resample_randomly = None
    # Utility class.
    mfm = None
    # Ongoing state.
    particle_set = None
    particle_weights = None
    # Filter output.
    best_weight = 0
    best_estimate = None

    def __init__(self):
        """
        Instantiate the particle filter and set params from the config yaml.
        """
        config = load_config()
        self.num_particles = int(config["particle_filter"]["num_particles"])
        self.all_indices = list(range(self.num_particles))
        self.state_size = int(config["particle_filter"]["state_size"])
        random_sampling_rate = config["particle_filter"]["random_sampling_rate"]
        self.num_to_resample_randomly = int(random_sampling_rate * self.num_particles)
        # Random particles are only injected when the best particle explains the observation
        # worse than this likelihood (robot lost / kidnapped). Injecting them every step lets
        # them take over the population whenever observations are ambiguous.
        self.random_sampling_threshold = float(config["particle_filter"].get("random_sampling_threshold", 0.5))
        # Gaussian noise added to resampled particles so the population doesn't collapse.
        self.resample_noise_xy = float(config["particle_filter"].get("resample_noise_xy", 0.02))
        self.resample_noise_yaw = float(config["particle_filter"].get("resample_noise_yaw", 0.02))
        # Particles within this radius (m) of each other form one hypothesis for the estimate.
        self.cluster_radius = float(config["particle_filter"].get("cluster_radius", 0.3))

        # Init arrays with correct dimensions.
        self.particle_set = np.zeros((self.num_particles, self.state_size))
        self.particle_weights = np.zeros(self.num_particles)
        self.best_estimate = np.zeros(self.state_size)

    def set_map_frame_manager(self, mfm: MapFrameManager):
        """
        Set reference to the map frame manager for coordinate transforms.
        @param mfm - MapFrameManager instance already initialized with a map.
        """
        self.mfm = mfm

    def propagate_particles(self, fwd: float, ang: float):
        """
        Apply a relative motion to all particles.
        @param fwd - Commanded forward motion in meters.
        @param ang - Commanded angular motion in radians (CCW).
        """
        for i in range(self.num_particles):
            self.particle_set[i, 0] += fwd * cos(self.particle_set[i, 2])
            self.particle_set[i, 1] += fwd * sin(self.particle_set[i, 2])
            # Keep yaw normalized to (-pi, pi).
            self.particle_set[i, 2] = remainder(self.particle_set[i, 2] + ang, tau)

        # Propagate the overall filter estimate as well.
        if self.best_estimate is not None:
            self.best_estimate[0] += fwd * cos(self.best_estimate[2])
            self.best_estimate[1] += fwd * sin(self.best_estimate[2])
            self.best_estimate[2] = remainder(self.best_estimate[2] + ang, tau)

    def update_with_observation(self, observation) -> PoseMeters:
        """
        Use an observation to evaluate particle likelihoods and update the filter estimate.
        @param observation - 2D numpy array of the observation for this iteration.
        @return PoseMeters of best particle estimate (x, y, yaw).
        """
        if observation is not None:
            for i in range(self.num_particles):
                obs_img_expected, _ = self.mfm.extract_observation_region(
                    PoseMeters(self.particle_set[i, 0], self.particle_set[i, 1], self.particle_set[i, 2])
                )
                self.particle_weights[i] = self.compute_measurement_likelihood(obs_img_expected, observation)
                # NOTE likelihoods are intentionally NOT normalized.

            # Re-estimate from THIS iteration's weights. (Comparing against the best weight
            # ever seen froze the estimate once one perfect match occurred.)
            self.best_estimate = self.cluster_estimate()
            self.best_weight = float(np.max(self.particle_weights))

        return PoseMeters(self.best_estimate[0], self.best_estimate[1], self.best_estimate[2])

    def cluster_estimate(self):
        """
        Weighted mean of the strongest particle cluster.

        Coarse observations often give many particles the same likelihood (e.g. every
        particle in an open corridor sees "all free"), so taking argmax of the weights
        would jump to an arbitrary particle anywhere on the map. Instead, pick the
        particle with the most weighted support within cluster_radius and average
        that neighbourhood.
        @return new numpy array (x, y, yaw) - a copy, never a view into particle_set.
        """
        w = np.asarray(self.particle_weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        xy = self.particle_set[:, :2]
        near = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2) <= self.cluster_radius ** 2
        support = w * (near @ w)
        members = near[int(np.argmax(support))] & (w > 0)
        mw = w[members]
        pts = self.particle_set[members]
        x = float(np.sum(pts[:, 0] * mw) / mw.sum())
        y = float(np.sum(pts[:, 1] * mw) / mw.sum())
        yaw = float(np.arctan2(np.sum(np.sin(pts[:, 2]) * mw), np.sum(np.cos(pts[:, 2]) * mw)))
        return np.array([x, y, yaw])

    def compute_measurement_likelihood(self, obs_expected, obs_actual) -> float:
        """
        Determine the likelihood of a specific particle given expected vs actual observations.
        @param obs_expected - 2D numpy array of the expected observation for a given particle.
        @param obs_actual   - 2D numpy array of the actual observation this iteration.
        @return float - likelihood of this particle.
        """
        # Kill particles that failed to generate an observation (too close to map edge).
        if obs_expected is None:
            return 0.0

        likelihood = 1.0
        for i in range(obs_expected.shape[0]):
            for j in range(obs_expected.shape[1]):
                diff = abs(obs_expected[i, j] - obs_actual[i, j])
                likelihood *= (1.0 - diff)
        return likelihood

    def resample(self):
        """
        Use the weights vector to sample from the population and form the next generation.
        """
        new_particle_set = np.zeros((self.num_particles, self.state_size))

        # Inject random particles only if the current population no longer explains the observation.
        observation_explained = float(np.max(self.particle_weights)) >= self.random_sampling_threshold
        num_random = 0 if observation_explained else self.num_to_resample_randomly

        # Ensure weights vector is not all zeros.
        if sum(self.particle_weights) == 0:
            self.particle_weights = np.ones(len(self.particle_weights))

        # Sample weighted particles to form most of the new population.
        selected_indices = choices(
            self.all_indices,
            list(self.particle_weights),
            k=self.num_particles - num_random
        )
        for i_new, i_old in enumerate(selected_indices):
            new_particle_set[i_new, :] = self.particle_set[i_old, :]
        # Perturb the resampled particles with noise to keep the population diverse.
        n_sel = len(selected_indices)
        new_particle_set[:n_sel, 0:2] += np.random.normal(0.0, self.resample_noise_xy, (n_sel, 2))
        new_particle_set[:n_sel, 2] += np.random.normal(0.0, self.resample_noise_yaw, n_sel)
        new_particle_set[:n_sel, 2] = (new_particle_set[:n_sel, 2] + np.pi) % (2 * np.pi) - np.pi

        # Randomly generate a small portion of the population to prevent particle depletion.
        for i in range(self.num_particles - num_random, self.num_particles):
            if self.mfm.initialized:
                new_particle_set[i, :] = self.mfm.generate_random_valid_veh_pose().as_np_array()
            else:
                new_particle_set[i, :] = np.zeros(self.state_size)

        self.particle_set = new_particle_set

    def get_particle_set_px(self):
        """
        Convert the particle set to a list of PosePixels for visualization.
        @return List of PosePixels.
        """
        return [
            self.mfm.transform_pose_m_to_px(
                PoseMeters(self.particle_set[i, 0], self.particle_set[i, 1], self.particle_set[i, 2])
            )
            for i in range(self.num_particles)
        ]