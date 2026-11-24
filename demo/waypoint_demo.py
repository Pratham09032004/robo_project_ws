#!/usr/bin/env python3
"""
Waypoint demo for the real VineBot - no map, no localization, no ML.

Wherever the robot is when this script starts is (0, 0), facing +x. The script makes a path through a few
short waypoints given relative to that start pose, publishes it for RViz / motion_confirm, and drives it
with the wheel odometry: turn on the spot towards the next waypoint, then drive to it while steering.

Run (after real_robot.launch.py; do NOT run autonomy.launch.py / runner_node / motion_planner at the same time):

    source /opt/ros/humble/setup.bash && source ~/robo_project_ws/install/setup.bash
    python3 waypoint_demo.py                          # default: 0.5 m square, back to the start
    python3 waypoint_demo.py --ros-args -p waypoints:="[0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]"

Topics
  in   odom_topic (default /odom)  nav_msgs/Odometry from the Roboteq controller
  out  /cmd_vel                    geometry_msgs/Twist -> motion_confirm -> twist_mux -> motors
       /planned_path               nav_msgs/Path (frame odom) - RViz, and motion_confirm asks "Move? [y/N]"
       /demo/waypoints             geometry_msgs/PoseArray (frame odom) - the waypoints, for RViz
       /runner_current_waypoint    geometry_msgs/PoseStamped - the waypoint the robot is driving to now

Safety
  - Slow (0.15 m/s, 0.35 rad/s by default). PS4 L1 (joystick) and O (emergency) still override everything.
  - Stops if the odometry stops (> 0.5 s), if a waypoint takes too long, and on Ctrl+C.
  - Stops if a turn goes the wrong way (heading error grows instead of shrinking): that means the turn
    direction of the motors / odometry is inverted - fix that before any autonomous driving.
"""
import math
import threading
import time

# ── control (no ROS - tested on its own) ──────────────────────────────────────


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def to_world(start, p):
    """Point p (x forward, y left of the start pose) -> odom frame."""
    sx, sy, sth = start
    c, s = math.cos(sth), math.sin(sth)
    return sx + c * p[0] - s * p[1], sy + s * p[0] + c * p[1]


def densify(points, step=0.05):
    """Straight segments through the points, one pose every `step` m (for the published path)."""
    out = [points[0]]
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        out += [(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(1, n + 1)]
    return out


class WaypointFollower:
    """Turn on the spot towards the waypoint, then drive to it with proportional steering."""

    def __init__(self, waypoints, lin=0.15, ang=0.35, min_ang=0.12, tol=0.08, turn_err=0.35, turn_exit=0.08,
                 wp_timeout=40.0):
        self.wps = list(waypoints)          # odom frame
        self.i = 0
        self.lin, self.ang, self.min_ang = lin, ang, min_ang
        self.tol, self.turn_err, self.turn_exit = tol, turn_err, turn_exit
        self.wp_timeout = wp_timeout
        self.turning = False
        self.turn_start_err = None
        self.wp_t0 = None
        self.done = False
        self.error = None                   # text when stopped because of a problem

    def step(self, x, y, yaw, t):
        """Pose (odom frame) and time -> (v, w). (0, 0) when done or stopped."""
        if self.done or self.error:
            return 0.0, 0.0
        if self.wp_t0 is None:
            self.wp_t0 = t
        wx, wy = self.wps[self.i]
        dx, dy = wx - x, wy - y
        dist = math.hypot(dx, dy)
        err = wrap(math.atan2(dy, dx) - yaw)
        # reached: close enough, or just driven past it (it is behind and close)
        if dist < self.tol or (dist < 2 * self.tol and abs(err) > math.pi / 2 and not self.turning):
            self.i += 1
            self.turning, self.turn_start_err, self.wp_t0 = False, None, t
            if self.i >= len(self.wps):
                self.done = True
            return 0.0, 0.0
        if t - self.wp_t0 > self.wp_timeout:
            self.error = f"waypoint {self.i + 1} not reached within {self.wp_timeout:.0f} s"
            return 0.0, 0.0
        if not self.turning and abs(err) > self.turn_err:
            self.turning, self.turn_start_err = True, abs(err)
        if self.turning:
            if abs(err) < self.turn_exit:
                self.turning, self.turn_start_err = False, None
            else:
                if abs(err) > self.turn_start_err + math.radians(30):
                    self.error = ("turn went the WRONG way (heading error grew from "
                                  f"{math.degrees(self.turn_start_err):.0f} to {math.degrees(abs(err)):.0f} deg): "
                                  "turn direction of motors / odometry is inverted")
                    return 0.0, 0.0
                self.turn_start_err = min(self.turn_start_err, abs(err))
                w = max(self.min_ang, min(self.ang, 1.2 * abs(err)))
                return 0.0, math.copysign(w, err)
        v = min(self.lin, max(0.06, dist))                    # slow down for the last few cm
        w = max(-self.ang, min(self.ang, 1.5 * err))
        return v, w


# ── ROS node ──────────────────────────────────────────────────────────────────


def main():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Twist
    from nav_msgs.msg import Odometry, Path

    def yaw_of(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    class WaypointDemo(Node):
        def __init__(self):
            super().__init__("waypoint_demo")
            p = self.declare_parameter
            # x, y pairs in metres relative to the start pose (x forward, y left). Default: 0.5 m square.
            self.rel = list(p("waypoints", [0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0]).value)
            self.odom_topic = str(p("odom_topic", "/odom").value)
            self.frame = str(p("frame", "odom").value)
            self.kw = dict(lin=float(p("lin_vel", 0.15).value), ang=float(p("ang_vel", 0.35).value),
                           min_ang=float(p("min_ang_vel", 0.12).value), tol=float(p("tolerance", 0.08).value))
            if len(self.rel) < 2 or len(self.rel) % 2:
                raise SystemExit("waypoints must be x, y pairs, e.g. [0.5, 0.0, 0.5, 0.5]")
            self.rel = [(float(self.rel[k]), float(self.rel[k + 1])) for k in range(0, len(self.rel), 2)]
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.path_pub = self.create_publisher(Path, "/planned_path", 1)
            self.wp_pub = self.create_publisher(PoseArray, "/demo/waypoints", 1)
            self.cur_pub = self.create_publisher(PoseStamped, "/runner_current_waypoint", 1)
            self.create_subscription(Odometry, self.odom_topic, self.on_odom, 20)
            self.pose = None
            self.pose_wall = 0.0
            self.start = None
            self.ctrl = None
            self.points = None
            self.finished = False
            self.last_log = 0.0
            self.create_timer(0.1, self.tick)
            self.create_timer(1.0, self.publish_path)
            self.get_logger().info(f"Waiting for odometry on {self.odom_topic} ...")

        def on_odom(self, m):
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
            self.pose_wall = time.monotonic()
            if self.start is None:
                self.start = self.pose
                pts = [to_world(self.start, q) for q in self.rel]
                self.points = [self.start[:2]] + pts
                self.ctrl = WaypointFollower(pts, **self.kw)
                route = "  ".join(f"({q[0]:.2f},{q[1]:.2f})" for q in self.rel)
                self.get_logger().info(f"Start pose = (0, 0, 0 deg)  [odom ({self.start[0]:.2f}, "
                                       f"{self.start[1]:.2f}, {math.degrees(self.start[2]):.0f} deg)]")
                self.get_logger().info(f"Plan: {len(pts)} waypoints relative to the start: {route}")
                self.get_logger().info("Path published on /planned_path. If motion_confirm runs, answer 'y' there.")
                self.publish_path()

        def stamped(self, x, y, th=0.0):
            ps = PoseStamped()
            ps.header.frame_id = self.frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            return ps

        def publish_path(self):
            if self.points is None:
                return
            path = Path()
            path.header.frame_id = self.frame
            path.header.stamp = self.get_clock().now().to_msg()
            path.poses = [self.stamped(x, y) for x, y in densify(self.points)]
            self.path_pub.publish(path)
            pa = PoseArray()
            pa.header = path.header
            for (ax, ay), (bx, by) in zip(self.points, self.points[1:]):
                q = Pose()
                q.position.x, q.position.y = float(bx), float(by)
                th = math.atan2(by - ay, bx - ax)
                q.orientation.z, q.orientation.w = math.sin(th / 2), math.cos(th / 2)
                pa.poses.append(q)
            self.wp_pub.publish(pa)

        def send(self, v, w):
            t = Twist()
            t.linear.x, t.angular.z = float(v), float(w)
            self.cmd_pub.publish(t)

        def tick(self):
            if self.ctrl is None or self.finished:
                return
            if time.monotonic() - self.pose_wall > 0.5:
                self.send(0.0, 0.0)
                self.get_logger().warn("No odometry for > 0.5 s - stopped.", throttle_duration_sec=2.0)
                return
            x, y, yaw = self.pose
            i_before = self.ctrl.i
            v, w = self.ctrl.step(x, y, yaw, time.monotonic())
            self.send(v, w)
            if self.ctrl.i != i_before:
                rx, ry = self.rel[i_before]
                self.get_logger().info(f"Waypoint {i_before + 1}/{len(self.rel)} reached ({rx:.2f}, {ry:.2f}).")
            if self.ctrl.done or self.ctrl.error:
                self.finished = True
                self.send(0.0, 0.0)
                sx, sy, sth = self.start
                c, s = math.cos(sth), math.sin(sth)
                lx, ly = c * (x - sx) + s * (y - sy), -s * (x - sx) + c * (y - sy)
                where = f"robot at ({lx:.2f}, {ly:.2f}, {math.degrees(wrap(yaw - sth)):.0f} deg) relative to the start"
                if self.ctrl.done:
                    self.get_logger().info(f"ALL {len(self.rel)} WAYPOINTS REACHED - stopped, {where}.")
                else:
                    self.get_logger().error(f"STOPPED: {self.ctrl.error}; {where}.")
                return
            wx, wy = self.ctrl.wps[self.ctrl.i]
            self.cur_pub.publish(self.stamped(wx, wy))
            now = time.monotonic()
            if now - self.last_log > 1.0:
                self.last_log = now
                rx, ry = self.rel[self.ctrl.i]
                self.get_logger().info(
                    f"-> waypoint {self.ctrl.i + 1}/{len(self.rel)} ({rx:.2f}, {ry:.2f}): dist "
                    f"{math.hypot(wx - x, wy - y):.2f} m, {'turning' if self.ctrl.turning else 'driving'}, "
                    f"cmd v={v:.2f} w={w:.2f}")

    rclpy.init()
    node = WaypointDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(3):
                node.send(0.0, 0.0)
                time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
