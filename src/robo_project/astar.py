#!/usr/bin/env python3

import numpy as np
from math import remainder, tau

try:
    import rclpy.logging
    from robo_project.scripts.basic_types import PosePixels, yaw_to_cardinal_dir, cardinal_dir_to_yaw
    _logger = rclpy.logging.get_logger('astar')
except ImportError:
    import logging
    _logger = logging.getLogger('astar')

    class PosePixels:
        def __init__(self, r, c, yaw=0.0):
            self.r = float(r)
            self.c = float(c)
            self.yaw = yaw

    yaw_to_cardinal_dir = {}
    cardinal_dir_to_yaw = {}


class Astar:
    verbose = False
    include_diagonals = False
    map = None  # 2D numpy array of the global map
    goal_cell: PosePixels = None

    last_path_px_reversed = None

    def run_astar(self, start_pose_px: PosePixels, goal_pose_px: PosePixels = None):
        """
        Use A* to generate a path from the current pose to the goal position.
        @return List of PosePixels describing the path in reverse (goal to start).
        """
        if self.map is None:
            _logger.error("A*: Cannot run_astar since self.map is None!")
            return
        start_cell = Cell(start_pose_px)
        if goal_pose_px is None:
            goal_pose_px = self.goal_cell
        goal_cell = Cell(goal_pose_px)

        if start_cell.out_of_bounds(self.map):
            _logger.error("A*: Starting position not within map bounds. Exiting without computing a path.")
            return
        if start_cell.in_collision(self.map):
            _logger.warning("A*: Starting position is in collision. Computing a path, and encouraging motion to free space.")
        if goal_cell.out_of_bounds(self.map):
            _logger.error("A*: Goal position not within map bounds. Exiting without computing a path.")
            return
        if goal_cell.in_collision(self.map):
            _logger.error("A*: Goal position is in collision. Exiting without computing a path.")
            return

        # FIX: nbrs was previously a class attribute computed ONCE at class
        # definition time, always using include_diagonals=False. Setting
        # planner.include_diagonals=True afterward never took effect —
        # the planner silently stayed cardinal-only regardless. Now
        # computed fresh here, using this instance's actual setting.
        nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)] + (
            [(-1, -1), (-1, 1), (1, -1), (1, 1)] if self.include_diagonals else [])

        open_list = [start_cell]
        closed_list = []
        while len(open_list) > 0:
            if self.verbose:
                _logger.info("A*: Iteration with len(open_list)={:}, len(closed_list)={:}".format(len(open_list), len(closed_list)))
            open_list.sort(key=lambda cell: cell.f)
            cur_cell = open_list.pop(0)
            if cur_cell == goal_cell:
                path_to_start = []
                while cur_cell.parent is not None:
                    path_to_start.append(PosePixels(cur_cell.r, cur_cell.c))
                    cur_cell = cur_cell.parent
                return path_to_start
            closed_list.append(cur_cell)
            for chg in nbrs:
                nbr = Cell(PosePixels(cur_cell.r + chg[0], cur_cell.c + chg[1]), parent=cur_cell)
                if nbr.r < 0 or nbr.c < 0 or nbr.r >= self.map.shape[0] or nbr.c >= self.map.shape[1]: continue
                if nbr.in_collision(self.map) and not nbr.parent_in_collision(self.map): continue
                if any([nbr == c for c in closed_list]): continue
                seen = [nbr == open_cell for open_cell in open_list]
                try:
                    match_i = seen.index(True)
                    if nbr.g < open_list[match_i].g:
                        open_list[match_i].set_cost(g=nbr.g)
                        open_list[match_i].parent = nbr.parent
                    continue
                except ValueError:
                    pass
                if self.include_diagonals:
                    nbr.set_cost(h=max(abs(goal_cell.r - nbr.r), abs(goal_cell.c - nbr.c)))
                else:
                    nbr.set_cost(h=(goal_cell.r - nbr.r) ** 2 + (goal_cell.c - nbr.c) ** 2)
                open_list.append(nbr)

    def get_next_discrete_action(self, start_pose_px: PosePixels) -> str:
        if self.map is None or self.goal_cell is None:
            _logger.error("A*: Cannot get_next_discrete_action unless self.map and self.goal_cell have been set!")
            raise RuntimeError("A* map or goal_cell not set.")
        self.include_diagonals = False
        self.last_path_px_reversed = self.run_astar(start_pose_px)
        if self.last_path_px_reversed is None or len(self.last_path_px_reversed) < 1:
            _logger.warning("A*: Unable to plan a path, so commanding a random discrete action.")
            return np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]
        next_cell = self.last_path_px_reversed[-1]
        dir_to_next_cell = start_pose_px.direction_to_cell(next_cell)
        dir_current_yaw = start_pose_px.get_direction()
        if dir_to_next_cell == dir_current_yaw:
            return "move_forward"
        yaw_diff_rads = remainder(cardinal_dir_to_yaw[dir_to_next_cell] - cardinal_dir_to_yaw[dir_current_yaw], tau)
        return "turn_left" if yaw_diff_rads > 0 else "turn_right"


class Cell:
    def __init__(self, pose_px: PosePixels, parent=None):
        if pose_px is None:
            _logger.error("A*: Illegal creation of Cell; pose_px cannot be None.")
            raise ValueError("Cell pose_px cannot be None.")
        self.r = int(pose_px.r)
        self.c = int(pose_px.c)
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0

    def out_of_bounds(self, map) -> bool:
        return self.r < 0 or self.c < 0 or self.r >= map.shape[0] or self.c >= map.shape[1]

    def in_collision(self, map) -> bool:
        return map[self.r, self.c] == 0

    def parent_in_collision(self, map) -> bool:
        if self.parent is None:
            return False
        return self.parent.in_collision(map)

    def set_cost(self, h=None, g=None, map=None):
        if h is not None:
            self.h = h
        if g is not None:
            self.g = g
        self.f = self.g + self.h
        if map is not None:
            if self.in_collision(map):
                self.f += 1000

    def __eq__(self, other):
        return self.r == other.r and self.c == other.c

    def __str__(self):
        return "Cell (" + str(self.r) + "," + str(self.c) + ") with costs " + str([self.g, self.f])
