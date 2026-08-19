import math
import numpy as np
from math import remainder, tau, hypot
from heapq import heappush, heappop

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
    verbose           = False
    include_diagonals = False
    map               = None
    goal_cell: PosePixels = None
    heuristic_map     = None
    last_path_px_reversed = None

    def _line_of_sight(self, p1: PosePixels, p2: PosePixels) -> bool:
        H, W = self.map.shape
        r0, c0 = float(p1.r), float(p1.c)
        r1, c1 = float(p2.r), float(p2.c)
        dist = hypot(r1 - r0, c1 - c0)
        if dist == 0:
            return True
        steps = int(dist / 0.1) + 1
        for i in range(steps + 1):
            t = i / steps
            r = r0 + t * (r1 - r0)
            c = c0 + t * (c1 - c0)
            for ri in [int(math.floor(r)), int(math.ceil(r))]:
                for ci in [int(math.floor(c)), int(math.ceil(c))]:
                    if not (0 <= ri < H and 0 <= ci < W):
                        return False
                    if self.map[ri, ci] == 0:
                        return False
        return True

    def smooth_path(self, path_reversed: list) -> list:
        if path_reversed is None or len(path_reversed) < 3:
            return path_reversed
        path_fwd = path_reversed[::-1]
        smoothed = [path_fwd[0]]
        i = 0
        while i < len(path_fwd) - 1:
            j = len(path_fwd) - 1
            while j > i + 1:
                if self._line_of_sight(path_fwd[i], path_fwd[j]):
                    break
                j -= 1
            smoothed.append(path_fwd[j])
            i = j
        return smoothed[::-1]

    def run_astar(self, start_pose_px: PosePixels, goal_pose_px: PosePixels = None):
        if self.map is None:
            _logger.error("A*: map is None!")
            return None

        start_cell = Cell(start_pose_px)
        if goal_pose_px is None:
            goal_pose_px = self.goal_cell
        goal_cell = Cell(goal_pose_px)

        if start_cell.out_of_bounds(self.map):
            _logger.error("A*: Start out of bounds.")
            return None
        if start_cell.in_collision(self.map):
            _logger.warning("A*: Start in collision.")
        if goal_cell.out_of_bounds(self.map):
            _logger.error("A*: Goal out of bounds.")
            return None
        if goal_cell.in_collision(self.map):
            _logger.error("A*: Goal in collision.")
            return None

        nbrs = [(0,-1,1.0),(0,1,1.0),(-1,0,1.0),(1,0,1.0)]
        if self.include_diagonals:
            nbrs += [(-1,-1,1.4),(-1,1,1.4),(1,-1,1.4),(1,1,1.4)]

        counter   = 0
        open_heap = []
        heappush(open_heap, (0.0, counter, start_cell))
        g_scores  = {(start_cell.r, start_cell.c): 0.0}
        closed_set = set()
        came_from  = {}

        while open_heap:
            _, _, cur_cell = heappop(open_heap)
            cur_key = (cur_cell.r, cur_cell.c)
            if cur_key in closed_set:
                continue
            closed_set.add(cur_key)

            if cur_cell == goal_cell:
                path = []
                curr = cur_cell
                while curr is not None:
                    path.append(PosePixels(curr.r, curr.c))
                    curr = came_from.get((curr.r, curr.c), None)
                return path

            for dr, dc, move_cost in nbrs:
                nr, nc  = cur_cell.r + dr, cur_cell.c + dc
                nbr_key = (nr, nc)

                if nr < 0 or nc < 0 or nr >= self.map.shape[0] or nc >= self.map.shape[1]:
                    continue
                if self.map[nr, nc] == 0 and self.map[cur_cell.r, cur_cell.c] != 0:
                    continue
                if dr != 0 and dc != 0:
                    if self.map[cur_cell.r+dr, cur_cell.c] == 0 or \
                       self.map[cur_cell.r, cur_cell.c+dc] == 0:
                        continue
                if nbr_key in closed_set:
                    continue

                new_g = g_scores[cur_key] + move_cost
                if new_g >= g_scores.get(nbr_key, float('inf')):
                    continue

                g_scores[nbr_key]  = new_g
                came_from[nbr_key] = cur_cell

                if self.heuristic_map is not None and \
                   0 <= nr < self.heuristic_map.shape[0] and \
                   0 <= nc < self.heuristic_map.shape[1]:
                    h = float(self.heuristic_map[nr, nc])
                    if np.isinf(h):
                        h = hypot(goal_cell.r - nr, goal_cell.c - nc)
                elif self.include_diagonals:
                    h = max(abs(goal_cell.r - nr), abs(goal_cell.c - nc))
                else:
                    h = hypot(goal_cell.r - nr, goal_cell.c - nc)

                nbr_cell   = Cell(PosePixels(nr, nc))
                nbr_cell.g = new_g
                nbr_cell.h = h
                nbr_cell.f = new_g + h
                counter += 1
                heappush(open_heap, (nbr_cell.f, counter, nbr_cell))

        return None

    def get_next_discrete_action(self, start_pose_px: PosePixels) -> str:
        if self.map is None or self.goal_cell is None:
            raise RuntimeError("A* map or goal_cell not set.")
        self.include_diagonals = False
        self.last_path_px_reversed = self.run_astar(start_pose_px)
        if self.last_path_px_reversed is None or len(self.last_path_px_reversed) < 1:
            _logger.warning("A*: No path — random action.")
            return np.random.choice(['move_forward','turn_left','turn_right'], 1)[0]
        next_cell        = self.last_path_px_reversed[-1]
        dir_to_next_cell = start_pose_px.direction_to_cell(next_cell)
        dir_current_yaw  = start_pose_px.get_direction()
        if dir_to_next_cell == dir_current_yaw:
            return "move_forward"
        yaw_diff_rads = remainder(
            cardinal_dir_to_yaw[dir_to_next_cell] - cardinal_dir_to_yaw[dir_current_yaw], tau)
        return "turn_left" if yaw_diff_rads > 0 else "turn_right"

    def get_smooth_next_action(self, start_pose_px: PosePixels) -> str:
        if self.map is None or self.goal_cell is None:
            raise RuntimeError("A* map or goal_cell not set.")
        self.include_diagonals = False
        raw_path = self.run_astar(start_pose_px)
        if raw_path is None or len(raw_path) < 1:
            return np.random.choice(['move_forward','turn_left','turn_right'], 1)[0]
        self.last_path_px_reversed = self.smooth_path(raw_path)
        next_cell        = self.last_path_px_reversed[-1]
        dir_to_next_cell = start_pose_px.direction_to_cell(next_cell)
        dir_current_yaw  = start_pose_px.get_direction()
        if dir_to_next_cell == dir_current_yaw:
            return "move_forward"
        yaw_diff_rads = remainder(
            cardinal_dir_to_yaw[dir_to_next_cell] - cardinal_dir_to_yaw[dir_current_yaw], tau)
        return "turn_left" if yaw_diff_rads > 0 else "turn_right"


class Cell:
    def __init__(self, pose_px: PosePixels):
        if pose_px is None:
            raise ValueError("Cell pose_px cannot be None.")
        self.r = int(pose_px.r)
        self.c = int(pose_px.c)
        self.g = 0
        self.h = 0
        self.f = 0

    def out_of_bounds(self, map) -> bool:
        return self.r < 0 or self.c < 0 \
               or self.r >= map.shape[0] or self.c >= map.shape[1]

    def in_collision(self, map) -> bool:
        return map[self.r, self.c] == 0

    def __eq__(self, other):
        return self.r == other.r and self.c == other.c

    def __lt__(self, other):
        return self.f < other.f

    def __str__(self):
        return "Cell ({},{}) g={:.1f} h={:.1f} f={:.1f}".format(
            self.r, self.c, self.g, self.h, self.f)
            
