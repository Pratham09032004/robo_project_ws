"""Discrete Bayes filter that localizes the robot directly on the coarse map
(paper Section IV-C).

State: one coarse-map cell (heading is assumed known, as in the paper).
Belief starts uniform over the free space of the map.

Predict ("noisy" transition, paper Eq. 1).  The paper moves the belief one
cell with probability p and keeps it with probability 1-p, with p ~ U[0, 1]
re-drawn every step because the map scale is unknown.  Here the robot moves
continuously, so the step is generalised: the odometry displacement d is
split into x / y cell fractions f = |d| / cell_size, and each axis moves the
belief one cell with probability p = clip(f * u, 0, 1), u ~ U[0, 2] (mean f,
random as in the paper).  Belief that would move into an occupied cell stays
put, like the paper's "space_to_wall" handling.

Coarse maps are misaligned with the world (the paper's motivation), so the
belief may also cover cells up to `support_dilation` cells outside the drawn
free space; the measurement still compares against the map as drawn.

Correct (paper Eq. 2-3).  The predicted local occupancy is resampled onto
the coarse grid around the robot (map axes, known heading) and compared to the
map window around every candidate cell:
    p(m | s) ∝ (1 - mean|F_psi(o) - m|) ** sharpness
sharpness = 1 is exactly the paper's similarity score.
"""

import math

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class CoarseMapBayesFilter:
    def __init__(self, free_fine: np.ndarray, fine_res: float, origin_xy, cell_size: float = 0.2,
                 window_radius: int = 7, sharpness: float = 1.0, uniform_floor: float = 1e-4,
                 support_dilation: int = 0, rng=None):
        """free_fine: bool array (row = +y) of the coarse map at fine_res m/px
        (e.g. the sketch from /map), origin_xy = world position of its corner."""
        self.cell = float(cell_size)
        self.R = int(window_radius)
        self.sharpness = float(sharpness)
        self.uniform_floor = float(uniform_floor)
        self.rng = rng or np.random.default_rng()
        self.origin = np.array(origin_xy, float)

        # Coarsen: a cell is free if at least half of it is free.
        k = max(1, int(round(self.cell / fine_res)))
        self.cell = k * fine_res
        h, w = free_fine.shape[0] // k, free_fine.shape[1] // k
        frac = free_fine[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))
        free = (frac >= 0.5).astype(np.uint8)
        # Keep the largest free region (the room); e.g. the white paper margin
        # outside a hand-drawn outer wall is "free" but not a place to be.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            free = (labels == big).astype(np.uint8)
        # The measurement compares against the map as drawn (occ) ...
        self.occ = 1.0 - free.astype(np.float32)
        # ... but the robot may be where the drawing says "wall": a coarse map is
        # misaligned with the world, so the belief may live up to
        # support_dilation cells outside the drawn free space.
        if support_dilation > 0:
            kern = np.ones((2 * support_dilation + 1,) * 2, np.uint8)
            free = cv2.dilate(free, kern)
        self.free = free.astype(bool)
        padded = np.pad(self.occ, self.R, constant_values=1.0)
        self._windows = sliding_window_view(padded, (2 * self.R + 1, 2 * self.R + 1))   # H, W, K, K
        self.reset()

    # ── belief ────────────────────────────────────────────────────────────────
    def reset(self):
        self.belief = self.free / self.free.sum()

    @property
    def shape(self):
        return self.free.shape

    def world_to_cell(self, x, y):
        return int((y - self.origin[1]) // self.cell), int((x - self.origin[0]) // self.cell)

    def cell_to_world(self, r, c):
        return self.origin[0] + (c + 0.5) * self.cell, self.origin[1] + (r + 0.5) * self.cell

    # ── predict (Eq. 1, generalised to continuous motion) ─────────────────────
    def _soft_shift(self, b, axis, step, p):
        if p <= 0.0 or step == 0:
            return b
        target_free = np.roll(self.free, -step, axis=axis)       # is my neighbour free?
        if axis == 0:
            target_free[(-1 if step > 0 else 0), :] = False
        else:
            target_free[:, (-1 if step > 0 else 0)] = False
        mover = b * p * target_free
        return b - mover + np.roll(mover, step, axis=axis)

    def predict(self, dx: float, dy: float):
        b = self.belief
        for axis, d in ((1, dx), (0, dy)):
            f = abs(d) / self.cell
            whole = int(f)
            step = 1 if d > 0 else -1
            for _ in range(whole):                       # displacements > 1 cell
                b = self._soft_shift(b, axis, step, 1.0)
            p = float(np.clip((f - whole) * self.rng.uniform(0.0, 2.0), 0.0, 1.0))
            b = self._soft_shift(b, axis, step, p)
        self.belief = b * self.free
        self._normalise()

    # ── correct (Eq. 2-3) ────────────────────────────────────────────────────
    def likelihood(self, local_window: np.ndarray) -> np.ndarray:
        """local_window: (2R+1)^2 map-aligned occupancy probabilities around the
        robot (NaN = unknown).  Returns p(m | s) for every cell."""
        valid = ~np.isnan(local_window)
        if valid.sum() == 0:
            return np.ones(self.shape)
        pred = np.where(valid, local_window, 0.0).astype(np.float32)
        diff = np.abs(self._windows - pred) * valid
        score = 1.0 - diff.sum(axis=(2, 3)) / valid.sum()
        return np.clip(score, 1e-6, 1.0) ** self.sharpness

    def correct(self, local_window: np.ndarray):
        self.belief = self.belief * self.likelihood(local_window) * self.free
        self._normalise()

    def _normalise(self):
        s = self.belief.sum()
        if not np.isfinite(s) or s <= 0:
            self.reset()
            return
        self.belief /= s
        if self.uniform_floor > 0:   # keeps every free cell recoverable
            self.belief = (1 - self.uniform_floor) * self.belief + self.uniform_floor * self.free / self.free.sum()

    # ── estimate ──────────────────────────────────────────────────────────────
    def estimate(self, conf_radius: int = 2):
        """(x, y, confidence, (r, c)): argmax cell (paper), refined to the
        belief-weighted mean of its 3x3 neighbourhood; confidence = belief mass
        within conf_radius cells of the argmax."""
        r, c = np.unravel_index(int(np.argmax(self.belief)), self.shape)
        r0, r1 = max(0, r - 1), min(self.shape[0], r + 2)
        c0, c1 = max(0, c - 1), min(self.shape[1], c + 2)
        patch = self.belief[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        rm, cm = (patch * rr).sum() / patch.sum(), (patch * cc).sum() / patch.sum()
        x, y = self.cell_to_world(rm, cm)
        conf = self.belief[max(0, r - conf_radius):r + conf_radius + 1,
                           max(0, c - conf_radius):c + conf_radius + 1].sum()
        return float(x), float(y), float(conf), (int(r), int(c))

    def entropy(self):
        b = self.belief[self.belief > 0]
        return float(-(b * np.log(b)).sum())


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
