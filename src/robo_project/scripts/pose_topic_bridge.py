import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point

class PoseTopicBridge(Node):
    def __init__(self):
        super().__init__("pose_topic_bridge")

        self.declare_parameter("input_topic", "/localization_pose_pf")
        self.declare_parameter("output_topic", "/ml_estimated_pose")

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        # Old full-simulation runner_node and pose_monitor_node expect Point
        self.pub = self.create_publisher(Point, self.output_topic, 10)

        # New PF localization publishes PoseStamped
        self.sub = self.create_subscription(
            PoseStamped,
            self.input_topic,
            self.pose_callback,
            10
        )

        self.get_logger().info(
            f"Pose bridge started: {self.input_topic} -> {self.output_topic} as geometry_msgs/Point"
        )

    def pose_callback(self, msg):
        out = Point()

        # runner_node reads Point.x and Point.y as the 2D map pose.
        out.x = float(msg.pose.position.x)
        out.y = float(msg.pose.position.y)
        out.z = 0.0

        self.pub.publish(out)

        self.get_logger().info(
            f"Forwarded PF pose: x={out.x:.3f}, y={out.y:.3f}"
        )

def main():
    rclpy.init()
    node = PoseTopicBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
