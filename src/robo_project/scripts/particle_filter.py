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

        # Find best particle this iteration.
        i_best = np.argmax(self.particle_weights)

        # Update filter estimate if this particle is better than the current best.
        if self.particle_weights[i_best] > self.best_weight:
            self.best_weight = self.particle_weights[i_best]
            self.best_estimate = self.particle_set[i_best, :]

        return PoseMeters(self.best_estimate[0], self.best_estimate[1], self.best_estimate[2])

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

        # Ensure weights vector is not all zeros.
        if sum(self.particle_weights) == 0:
            self.particle_weights = [1 for _ in range(len(self.particle_weights))]

        # Sample weighted particles to form most of the new population.
        selected_indices = choices(
            self.all_indices,
            list(self.particle_weights),
            k=self.num_particles - self.num_to_resample_randomly
        )
        for i_new, i_old in enumerate(selected_indices):
            new_particle_set[i_new, :] = self.particle_set[i_old, :]
            # TODO: Perturb with noise.

        # Randomly generate a small portion of the population to prevent particle depletion.
        for i in range(self.num_particles - self.num_to_resample_randomly, self.num_particles):
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