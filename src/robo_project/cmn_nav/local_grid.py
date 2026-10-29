"""Geometry of the egocentric local occupancy grid.

One convention is used everywhere (dataset labels, network output, ROS
messages, Bayes filter):

  * The local grid is N x N cells of LOCAL_RES metres, centred on the robot.
  * Row 0 is the far edge AHEAD of the robot, column 0 is the robot's LEFT
    (i.e. "robot-up" top-down image).
  * Values are the probability that the cell is OCCUPIED (1 = wall/obstacle).

World / map frame is the ROS one (REP-103): x right, y up, yaw CCW from +x.
Coarse-map arrays follow nav_msgs/OccupancyGrid: row index grows with +y.
"""

import math

import numpy as np

LOCAL_SIZE = 64        # cells
LOCAL_RES = 0.05       # m per cell  -> 3.2 m x 3.2 m window


def egocentric_offsets(n: int = LOCAL_SIZE, res: float = LOCAL_RES):
    """Forward / right offsets (m) of the cell centres: f[i], r[j]."""
    f = (n / 2.0 - np.arange(n) - 0.5) * res
    r = (np.arange(n) - n / 2.0 + 0.5) * res
    return f, r


def egocentric_world_points(x: float, y: float, yaw: float,
                            n: int = LOCAL_SIZE, res: float = LOCAL_RES):
    """World coordinates (X, Y), each n x n, of every egocentric cell centre."""
    f, r = egocentric_offsets(n, res)
    F, R = np.meshgrid(f, r, indexing="ij")
    c, s = math.cos(yaw), math.sin(yaw)
    # forward = (cos, sin), right = (sin, -cos)
    return x + F * c + R * s, y + F * s - R * c


def egocentric_index(dx, dy, yaw: float, n: int = LOCAL_SIZE, res: float = LOCAL_RES):
    """Egocentric (row, col) indices of world offsets (dx, dy) from the robot.
    Indices outside [0, n) mean the point is outside the local window."""
    c, s = math.cos(yaw), math.sin(yaw)
    f = dx * c + dy * s
    r = dx * s - dy * c
    i = np.floor(n / 2.0 - f / res).astype(int)
    j = np.floor(r / res + n / 2.0).astype(int)
    return i, j


def to_map_aligned(local_occ: np.ndarray, yaw: float, cell_size: float, radius_cells: int,
                   res: float = LOCAL_RES, subsamples: int = 4) -> np.ndarray:
    """Resample an egocentric local grid onto a (2R+1)^2 window of coarse-map
    cells, aligned with the map axes and centred on the robot's cell.

    Returns K x K occupancy probabilities, row index growing with +y; cells
    that are mostly outside the egocentric window are NaN (= no information).
    """
    n = local_occ.shape[0]
    k = 2 * radius_cells + 1
    # sub-sample points inside each coarse cell
    sub = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    cells = np.arange(k) - radius_cells
    off = (cells[:, None] + sub[None, :]).reshape(-1) * cell_size   # (k*m,)
    DY, DX = np.meshgrid(off, off, indexing="ij")
    i, j = egocentric_index(DX, DY, yaw, n, res)
    inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    vals = np.where(inside, local_occ[np.clip(i, 0, n - 1), np.clip(j, 0, n - 1)], 0.0)
    m = subsamples
    vals = vals.reshape(k, m, k, m).sum(axis=(1, 3))
    cnt = inside.reshape(k, m, k, m).sum(axis=(1, 3))
    out = np.full((k, k), np.nan, dtype=np.float32)
    ok = cnt >= (m * m) / 2
    out[ok] = vals[ok] / cnt[ok]
    return out


# ── nav_msgs/OccupancyGrid in the robot frame (base_footprint) ───────────────
# OccupancyGrid cell (row, col) is at x = origin + (col+.5)*res (forward),
# y = origin + (row+.5)*res (left).  Egocentric cell (i, j) is at
# forward = (n/2 - i - .5)*res, right = (j - n/2 + .5)*res, so
# grid[row, col] = local[n-1-col, n-1-row].

def egocentric_to_base_grid(local_occ: np.ndarray) -> np.ndarray:
    return local_occ[::-1, ::-1].T


def base_grid_to_egocentric(grid: np.ndarray) -> np.ndarray:
    return grid.T[::-1, ::-1]
