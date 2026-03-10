#!/usr/bin/env python3
"""
obstacle_avoid_node.py
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

WARN_MM   = 150.0
CRIT_MM   = 120.0
TARGET_MM = 125.0
WARN_M    = WARN_MM  / 1000.0
CRIT_M    = CRIT_MM  / 1000.0
TARGET_M  = TARGET_MM / 1000.0

ESCAPE_LIN = 0.12
ESCAPE_ANG = 0.6
LIDAR_OFFSET_DEG = 180.0


def ros_angle_to_robot_deg(lidar_angle_rad, offset_deg=0.0):
    deg = -math.degrees(lidar_angle_rad) + offset_deg
    return deg % 360.0


def robot_deg_to_vector(angle_deg):
    rad = math.radians(angle_deg)
    x =  math.cos(rad)
    y = -math.sin(rad)
    return x, y


class ObstacleAvoidNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoid_node')

        self.declare_parameter('warn_mm',          WARN_MM)
        self.declare_parameter('crit_mm',          CRIT_MM)
        self.declare_parameter('target_mm',        TARGET_MM)
        self.declare_parameter('escape_lin_speed', ESCAPE_LIN)
        self.declare_parameter('escape_ang_speed', ESCAPE_ANG)
        self.declare_parameter('lidar_offset_deg', LIDAR_OFFSET_DEG)

        self.warn_m    = self.get_parameter('warn_mm').value          / 1000.0
        self.crit_m    = self.get_parameter('crit_mm').value          / 1000.0
        self.target_m  = self.get_parameter('target_mm').value        / 1000.0
        self.esc_lin   = self.get_parameter('escape_lin_speed').value
        self.esc_ang   = self.get_parameter('escape_ang_speed').value
        self.lidar_off = self.get_parameter('lidar_offset_deg').value

        self.hand_cmd  = Twist()
        self.is_locked = False

        self.create_subscription(LaserScan, '/scan',          self.scan_callback, 10)
        self.create_subscription(Twist,     '/cmd_vel_raw',   self.hand_callback, 10)
        self.create_subscription(Bool,      '/motion_locked', self.lock_callback, 10)

        self.cmd_pub  = self.create_publisher(Twist,  '/cmd_vel',         10)
        self.warn_pub = self.create_publisher(String, '/obstacle_warning', 10)

        self.last_cmd = Twist()
        self.create_timer(0.05, self.publish_timer)

        self.get_logger().info("━━━ Obstacle Avoid Node Ready ━━━")
        self.get_logger().info(f"  WARNING  < {self.warn_m*1000:.0f} mm")
        self.get_logger().info(f"  CRITICAL < {self.crit_m*1000:.0f} mm")

    def hand_callback(self, scan_msg):
        self.hand_cmd = scan_msg

    def lock_callback(self, scan_msg):
        self.is_locked = scan_msg.data
        state = "LOCKED" if self.is_locked else "UNLOCKED"
        self.get_logger().info(f"[ObstacleAvoid] Motion {state}")

    def parse_scan(self, scan_msg):
        obstacles = []
        for i, r in enumerate(scan_msg.ranges):
            if not math.isfinite(r):
                continue
            if r < scan_msg.range_min or r > scan_msg.range_max:
                continue
            lidar_rad = scan_msg.angle_min + i * scan_msg.angle_increment
            robot_deg = ros_angle_to_robot_deg(lidar_rad, self.lidar_off)
            r_real = r * 1.54
            obstacles.append((robot_deg, r_real))
        return obstacles

    def compute_escape_vector(self, crit_obs):
        fx, fy = 0.0, 0.0
        for angle_deg, dist_m in crit_obs:
            obs_x, obs_y = robot_deg_to_vector(angle_deg)
            escape_x = -obs_x
            escape_y = -obs_y
            weight = max(0.0, (self.target_m - dist_m) / self.target_m) ** 2
            fx += escape_x * weight
            fy += escape_y * weight
        mag = math.hypot(fx, fy)
        if mag < 1e-6:
            return -1.0, 0.0
        return fx / mag, fy / mag

    def compute_escape_cmd(self, crit_obs, closest_deg, closest_m):
        twist = Twist()
        fx, fy = self.compute_escape_vector(crit_obs)
        error  = max(0.0, self.target_m - closest_m)
        scale  = max(0.3, min(1.0, error / (self.target_m - self.crit_m + 1e-6)))
        twist.linear.x = fx * self.esc_lin * scale
        twist.linear.y = fy * self.esc_lin * scale
        ang_err = closest_deg if closest_deg <= 180 else closest_deg - 360
        if 45 < abs(ang_err) < 135:
            ang_scale = min(1.0, abs(ang_err) / 90.0)
            twist.angular.z = math.copysign(self.esc_ang * ang_scale * 0.5, ang_err)
        return twist

    def publish_timer(self):
        if not self.is_locked:
            self.cmd_pub.publish(self.last_cmd)

    def scan_callback(self, scan_msg):
        if self.is_locked:
            self.last_cmd = Twist()
            return

        all_obs  = self.parse_scan(scan_msg)
        warn_obs = [(a, d) for a, d in all_obs if d < self.warn_m]
        crit_obs = [(a, d) for a, d in all_obs if d < self.crit_m]

        # ── Find closest among ALL obstacles in warn zone ─────────
        # FIX: define closest_deg/closest_m from crit_obs if warn_obs empty
        if crit_obs:
            closest     = min(crit_obs, key=lambda x: x[1])
            closest_deg = closest[0]
            closest_mm  = closest[1] * 1000.0
        elif warn_obs:
            closest     = min(warn_obs, key=lambda x: x[1])
            closest_deg = closest[0]
            closest_mm  = closest[1] * 1000.0
        else:
            # Nothing nearby — safe, publish SAFE and pass hand cmd
            w = String()
            w.data = "SAFE|0|0"
            self.warn_pub.publish(w)
            self.last_cmd = self.hand_cmd
            return

        # ── Log + publish warning ─────────────────────────────────
        if crit_obs:
            self.get_logger().error(
                f"[CRITICAL] Closest: {closest_mm:.0f}mm @ {closest_deg:.1f}deg"
            )
            w = String()
            w.data = f"CRITICAL|{closest_mm:.0f}|{closest_deg:.1f}"
            self.warn_pub.publish(w)
        else:
            self.get_logger().warn(
                f"[WARNING]  Closest: {closest_mm:.0f}mm @ {closest_deg:.1f}deg"
            )
            w = String()
            w.data = f"WARNING|{closest_mm:.0f}|{closest_deg:.1f}"
            self.warn_pub.publish(w)

        # ── Escape or pass through ────────────────────────────────
        if crit_obs:
            self.last_cmd = self.compute_escape_cmd(
                crit_obs, closest_deg, closest[1]
            )
        else:
            self.last_cmd = self.hand_cmd


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
