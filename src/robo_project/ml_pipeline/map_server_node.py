#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import numpy as np

# Import your existing processor layout directly
from robo_project.scripts.map_handler import MapFrameManager


def build_occupancy_grid_msg(map_with_border: np.ndarray, resolution: float) -> OccupancyGrid:
    """
    Convert the MapFrameManager grid into a ROS OccupancyGrid that lines up
    with MapFrameManager.transform_map_px_to_m()/transform_map_m_to_px().

    Internal map: image rows (row 0 = top), 1 = free, anything else = occupied.
    MapFrameManager world coords: x = res * (col - W//2), y = -res * (row - H//2).
    ROS OccupancyGrid: row 0 is at origin.y (bottom), 0 = free, 100 = occupied.
    So the image is flipped vertically and the origin is chosen such that
    floor((y - origin_y) / res) == H - 1 - row for every map cell.
    """
    height, width = map_with_border.shape
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = float(resolution)
    msg.info.width = int(width)
    msg.info.height = int(height)
    msg.info.origin.position.x = float(-(width // 2) * resolution)
    msg.info.origin.position.y = float(-(height - 1 - height // 2) * resolution)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    ros_rows = np.flipud(map_with_border)
    msg.data = np.where(ros_rows == 1, 0, 100).astype(np.int8).flatten().tolist()
    return msg


class StaticMapServerNode(Node):
    """
    The single publisher of /map and of the static map -> odom transform.
    """

    def __init__(self):
        super().__init__('static_map_server_node')

        self.map_manager = MapFrameManager(use_discrete_state_space=True)

        # Latched map: late subscribers (RViz, motion_planner) get it immediately.
        # TRANSIENT_LOCAL publishers are also compatible with VOLATILE subscribers.
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.map_msg = build_occupancy_grid_msg(
            self.map_manager.map_with_border, self.map_manager.map_resolution_desired)

        self.broadcast_static_transforms()
        self.publish_map()
        # Republish periodically for tools that subscribe with VOLATILE durability.
        self.timer = self.create_timer(1.0, self.publish_map)
        self.get_logger().info(
            f"Static map server publishing /map: {self.map_msg.info.width}x"
            f"{self.map_msg.info.height} @ {self.map_msg.info.resolution} m/cell")

    def broadcast_static_transforms(self):
        # Habitat odometry is already expressed in the map frame, so map -> odom is identity.
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StaticMapServerNode()
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
