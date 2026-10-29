#!/usr/bin/env python3
"""ROS 2 node: 4 Habitat cameras -> predicted egocentric local occupancy.

Subscribes  /camera/{front,left,back,right}/image_raw  (bgr8, from the bridge)
Publishes   /cmn/local_occ  nav_msgs/OccupancyGrid in base_footprint
            (0-100 = probability occupied; shows up around the robot in RViz)
"""

import os
import time

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image

from robo_project.cmn_nav.local_grid import LOCAL_RES, LOCAL_SIZE, egocentric_to_base_grid
from robo_project.cmn_nav.local_occ_model import images_to_tensor, load_model

MAX_VIEW_SPREAD = 0.3   # s - the 4 views must arrive this close together (one render)


class LocalOccNode(Node):
    def __init__(self):
        super().__init__('local_occ_node')
        self.declare_parameter('model_path', '')   # '' = the model installed with the package
        self.declare_parameter('rate_hz', 4.0)
        self.declare_parameter('threads', 2)
        torch.set_num_threads(int(self.get_parameter('threads').value))

        path = self.get_parameter('model_path').value or os.path.join(
            get_package_share_directory('robo_project'), 'ml_runs', 'cmn_local_occ.pth')
        self.model, ckpt = load_model(path)
        self.views = list(ckpt['views'])
        self.img_hw = tuple(ckpt['model_args']['img_hw'])
        assert ckpt['local_size'] == LOCAL_SIZE and abs(ckpt['local_res'] - LOCAL_RES) < 1e-9
        self.get_logger().info(f"Loaded {path} (epoch {ckpt.get('epoch')}, test {ckpt.get('test')})")

        self.bridge = CvBridge()
        self.latest = {}   # view -> (arrival time, RGB image at network size)
        for v in self.views:
            self.create_subscription(Image, f'/camera/{v}/image_raw',
                                     lambda msg, v=v: self.image_cb(v, msg), 1)
        self.pub = self.create_publisher(OccupancyGrid, '/cmn/local_occ', 1)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self.tick)
        self._n = 0
        self._last_used = 0.0

    def image_cb(self, view, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # same preprocessing as generate_dataset.preprocess_view()
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.img_hw[1], self.img_hw[0]),
                         interpolation=cv2.INTER_AREA)
        self.latest[view] = (time.monotonic(), rgb)

    def tick(self):
        if len(self.latest) < len(self.views):
            return
        stamps = [self.latest[v][0] for v in self.views]
        if min(stamps) <= self._last_used or max(stamps) - min(stamps) > MAX_VIEW_SPREAD:
            return                       # no new complete panorama yet
        self._last_used = max(stamps)
        pano = np.stack([self.latest[v][1] for v in self.views])[None]
        with torch.no_grad():
            prob = torch.sigmoid(self.model(images_to_tensor(pano)))[0].numpy()

        msg = OccupancyGrid()
        msg.header.frame_id = 'base_footprint'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = LOCAL_RES
        msg.info.width = msg.info.height = LOCAL_SIZE
        msg.info.origin.position.x = msg.info.origin.position.y = -LOCAL_SIZE * LOCAL_RES / 2.0
        msg.info.origin.orientation.w = 1.0
        msg.data = np.round(egocentric_to_base_grid(prob) * 100).astype(np.int8).flatten().tolist()
        self.pub.publish(msg)
        self._n += 1
        if self._n == 1:
            self.get_logger().info("Publishing predicted local occupancy on /cmn/local_occ")


def main(args=None):
    rclpy.init(args=args)
    node = LocalOccNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
