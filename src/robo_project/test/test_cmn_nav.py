"""Unit tests for the CMN geometry and Bayes filter (no ROS, no Habitat).

    python3 -m pytest src/robo_project/test/test_cmn_nav.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cmn_nav.bayes_filter import CoarseMapBayesFilter                                      # noqa: E402
from cmn_nav.local_grid import (LOCAL_SIZE, base_grid_to_egocentric, egocentric_index,     # noqa: E402
                                egocentric_to_base_grid, egocentric_world_points, to_map_aligned)


def test_egocentric_axes():
    # robot at (2, 3) facing +y: "ahead" is +y, "right" is +x
    i, j = egocentric_index(np.array(0.0), np.array(1.0), math.pi / 2)
    X, Y = egocentric_world_points(2.0, 3.0, math.pi / 2)
    assert i < LOCAL_SIZE / 2 and abs(X[i, j] - 2.0) < 0.05 and abs(Y[i, j] - 4.0) < 0.05
    i, j = egocentric_index(np.array(1.0), np.array(0.0), math.pi / 2)
    assert j > LOCAL_SIZE / 2


def test_base_grid_round_trip():
    p = np.random.default_rng(0).random((LOCAL_SIZE, LOCAL_SIZE))
    assert np.array_equal(base_grid_to_egocentric(egocentric_to_base_grid(p)), p)


def test_map_aligned_window_rotates_with_heading():
    occ = np.zeros((LOCAL_SIZE, LOCAL_SIZE), np.float32)
    occ[10:14, :] = 1.0                                   # wall ~1 m ahead
    w = to_map_aligned(occ, math.pi / 2, 0.2, 7)          # facing +y -> wall at +y rows
    assert w[7 + 5, 7] == 1.0 and w[7 - 5, 7] == 0.0
    w = to_map_aligned(occ, 0.0, 0.2, 7)                  # facing +x -> wall at +x cols
    assert w[7, 7 + 5] == 1.0 and w[7, 7 - 5] == 0.0


def _room():
    """20 x 14 m room at 0.02 m/px with a few distinct obstacles (row = +y)."""
    free = np.ones((700, 1000), bool)
    free[:25, :] = free[-25:, :] = free[:, :25] = free[:, -25:] = False
    free[150:300, 200:260] = False
    free[400:450, 500:800] = False
    free[100:200, 650:700] = False
    free[500:680, 150:200] = False
    return free


def test_filter_localizes_with_consistent_observations():
    free = _room()
    res, origin = 0.02, (0.0, 0.0)
    rng = np.random.default_rng(3)
    f = CoarseMapBayesFilter(free, res, origin, cell_size=0.2, window_radius=7, sharpness=2.0, rng=rng)
    assert abs(f.belief.sum() - 1.0) < 1e-9

    def observe(x, y, yaw):
        X, Y = egocentric_world_points(x, y, yaw)
        r = np.clip((Y / res).astype(int), 0, free.shape[0] - 1)
        c = np.clip((X / res).astype(int), 0, free.shape[1] - 1)
        return (~free[r, c]).astype(np.float32)

    x, y, yaw = 1.0, 2.6, 0.0                              # passes just below an obstacle
    for _ in range(50):                                    # drive +x
        f.correct(to_map_aligned(observe(x, y, yaw), yaw, f.cell, f.R))
        f.predict(0.15, 0.0)
        x += 0.15
    ex, ey, conf, _ = f.estimate()
    assert math.hypot(ex - x, ey - y) < 0.6, (ex, ey, x, y)
    assert conf > 0.5


def test_predict_does_not_enter_walls():
    free = _room()
    f = CoarseMapBayesFilter(free, 0.02, (0.0, 0.0), cell_size=0.2, uniform_floor=0.0,
                             rng=np.random.default_rng(0))
    for _ in range(200):
        f.predict(0.0, -0.2)                               # push everything into the y=0 wall
    assert np.all(f.belief[~f.free] == 0)
    assert abs(f.belief.sum() - 1.0) < 1e-6
