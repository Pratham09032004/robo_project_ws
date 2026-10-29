#!/usr/bin/env python3
"""Render a local-occupancy training set from Habitat (paper Section IV-B).

Each sample is a 4-view panorama (front, left, back, right) rendered with the
SAME camera rig as habitat_bridge_vinebot_2.py (0.6 m high, 90 deg HFOV, 4:3)
plus the ground-truth egocentric local occupancy around the robot, taken from
the scene's navmesh (navigable floor = free, everything else = occupied).

Runs in the conda `habitat` env (needs habitat_sim, no ROS):

    conda activate habitat
    python src/robo_project/cmn_nav/generate_dataset.py \
        --scene src/environment/IMR_lab8.glb --samples 6000 --out data/imr_lab8.npz
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import habitat_sim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_grid import LOCAL_RES, LOCAL_SIZE, egocentric_world_points  # noqa: E402

VIEWS = ("front", "left", "back", "right")   # order of the panorama channels
VIEW_YAW = {"front": 0.0, "right": -math.pi / 2, "back": math.pi, "left": math.pi / 2}
CAM_HEIGHT = 0.6
RENDER_HW = (240, 320)      # same 4:3 aspect + HFOV as the bridge's 480x640
IMG_HW = (72, 96)           # stored / network input size per view
FLOOR_MAX_Y = 0.5           # navmesh points above this are desks / roof, not floor
YAW_OFFSET = math.pi / 2    # ros_yaw = habitat_yaw + pi/2 (see the bridge)


def make_sim(scene: str, gpu: int):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.gpu_device_id = gpu
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    specs = []
    for name in VIEWS:
        cam = habitat_sim.CameraSensorSpec()
        cam.uuid = name
        cam.sensor_type = habitat_sim.SensorType.COLOR
        cam.resolution = list(RENDER_HW)
        cam.position = [0.0, CAM_HEIGHT, 0.0]
        cam.orientation = [0.0, VIEW_YAW[name], 0.0]
        specs.append(cam)
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def preprocess_view(rgb: np.ndarray) -> np.ndarray:
    """Exactly what local_occ_node does to a camera image (RGB in, RGB out)."""
    return cv2.resize(rgb[:, :, :3], (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1, help="-1 = software rendering")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = make_sim(args.scene, args.gpu)
    pf = sim.pathfinder
    pf.seed(args.seed)
    agent = sim.initialize_agent(0)

    # Ground truth: navigable floor, top-down at LOCAL_RES.
    floor_y = None
    for _ in range(1000):
        p = pf.get_random_navigable_point()
        if p[1] <= FLOOR_MAX_Y:
            floor_y = float(p[1])
            break
    lo, _ = pf.get_bounds()
    nav = pf.get_topdown_view(LOCAL_RES, floor_y).astype(np.uint8)   # rows: z, cols: x

    images = np.zeros((args.samples, len(VIEWS), IMG_HW[0], IMG_HW[1], 3), np.uint8)
    occ = np.zeros((args.samples, LOCAL_SIZE, LOCAL_SIZE), np.uint8)
    poses = np.zeros((args.samples, 3), np.float32)   # ROS x, y, yaw

    t0 = time.time()
    n = 0
    while n < args.samples:
        p = pf.get_random_navigable_point()
        if p[1] > FLOOR_MAX_Y:
            continue
        hab_yaw = rng.uniform(-math.pi, math.pi)
        state = agent.get_state()
        state.position = p
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            hab_yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        for v, name in enumerate(VIEWS):
            images[n, v] = preprocess_view(obs[name])

        x, y, yaw = float(p[0]), float(-p[2]), hab_yaw + YAW_OFFSET
        X, Y = egocentric_world_points(x, y, yaw)
        col = np.floor((X - lo[0]) / LOCAL_RES).astype(int)
        row = np.floor((-Y - lo[2]) / LOCAL_RES).astype(int)          # habitat z = -y
        inside = (row >= 0) & (row < nav.shape[0]) & (col >= 0) & (col < nav.shape[1])
        free = np.zeros_like(inside)
        free[inside] = nav[row[inside], col[inside]] > 0
        occ[n] = (~free).astype(np.uint8)
        poses[n] = (x, y, math.atan2(math.sin(yaw), math.cos(yaw)))
        n += 1
        if n % 500 == 0:
            print(f"{n}/{args.samples}  {n / (time.time() - t0):.1f} samples/s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, images=images, occ=occ, poses=poses,
                        views=np.array(VIEWS), scene=os.path.basename(args.scene))
    print(f"saved {n} samples to {args.out}")


if __name__ == "__main__":
    main()
