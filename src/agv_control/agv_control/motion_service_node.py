#!/usr/bin/env python3
"""
motion_service_node.py
======================
Node : motion_service_node
Sub  : /cmd_vel_raw   (geometry_msgs/Twist)  <- hand gesture
Pub  : /motion_locked (std_msgs/Bool)        <- lock status to obstacle node
       /cmd_vel       (geometry_msgs/Twist)  <- direct cmd when LOCKED

Service: /motion_control (agv_control_msgs/srv/MotionControl)

Priority flow:
  UNLOCKED: hand -> /cmd_vel_raw -> obstacle_avoid -> /cmd_vel -> robot
  LOCKED:   service -> [this node] -> /cmd_vel -> robot  (bypass everything)

When LOCKED:
  - publishes /motion_locked = True
  - obstacle_avoid_node sees this and stops forwarding anything
  - this node has full control of /cmd_vel
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from agv_control_msgs.srv import MotionControl


class MotionServiceNode(Node):

    def __init__(self):
        super().__init__('motion_service_node')

        self.is_locked   = False
        self.current_vel = Twist()

        # Subscribe hand gesture (only used to monitor, not forward)
        self.create_subscription(Twist, '/cmd_vel_raw', self.hand_callback, 10)

        # Publish lock status -> obstacle_avoid_node listens to this
        self.lock_pub = self.create_publisher(Bool,  '/motion_locked', 10)

        # Publish direct cmd when locked
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel',       10)

        # Service server
        self.srv = self.create_service(
            MotionControl,
            '/motion_control',
            self.handle_motion_control
        )

        # Timer to keep publishing lock status (so new nodes get it)
        self.create_timer(1.0, self.publish_lock_status)

        self.get_logger().info("━━━ Motion Service Node Ready ━━━")
        self.get_logger().info("  Service  : /motion_control")
        self.get_logger().info("  Commands : LOCK | UNLOCK | MOVE | STOP | STATUS")

    # ══════════════════════════════════════════════════════════

    def hand_callback(self, msg):
        """Monitor hand cmd but don't forward — obstacle node does that."""
        self.current_vel = msg

    def publish_lock_status(self):
        msg = Bool()
        msg.data = self.is_locked
        self.lock_pub.publish(msg)

    def publish_stop(self):
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.current_vel = stop

    # ══════════════════════════════════════════════════════════

    def handle_motion_control(self, request, response):
        cmd = request.command.upper().strip()
        self.get_logger().info(f"[Service] Command: '{cmd}'")

        # ── LOCK ─────────────────────────────────────────────
        if cmd == "LOCK":
            self.is_locked = True
            self.publish_lock_status()
            self.publish_stop()
            response.success = True
            response.message = "LOCKED. Hand input blocked. Use MOVE to control robot."
            self.get_logger().warn("━ Human input LOCKED ━")

        # ── UNLOCK ───────────────────────────────────────────
        elif cmd == "UNLOCK":
            self.is_locked = False
            self.publish_lock_status()
            response.success = True
            response.message = "UNLOCKED. Hand gesture control restored."
            self.get_logger().info("━ Human input UNLOCKED ━")

        # ── MOVE ─────────────────────────────────────────────
        elif cmd == "MOVE":
            if not self.is_locked:
                response.success = False
                response.message = "Send LOCK command first before MOVE."
            else:
                vel = Twist()
                vel.linear.x  = float(request.linear_x)
                vel.linear.y  = float(request.linear_y)
                vel.angular.z = float(request.angular_z)
                self.cmd_pub.publish(vel)
                self.current_vel = vel

                if request.duration_sec > 0.0:
                    self.create_timer(request.duration_sec, self.timed_stop)
                    self.get_logger().info(
                        f"Moving {request.duration_sec:.1f}s: "
                        f"vx={vel.linear.x:.2f} vy={vel.linear.y:.2f} "
                        f"wz={vel.angular.z:.2f}"
                    )
                else:
                    self.get_logger().info(
                        f"Moving continuous: "
                        f"vx={vel.linear.x:.2f} vy={vel.linear.y:.2f} "
                        f"wz={vel.angular.z:.2f}"
                    )

                response.success = True
                response.message = (
                    f"Moving: vx={vel.linear.x:.2f} "
                    f"vy={vel.linear.y:.2f} wz={vel.angular.z:.2f}"
                )

        # ── STOP ─────────────────────────────────────────────
        elif cmd == "STOP":
            self.publish_stop()
            response.success = True
            response.message = "Robot stopped."
            self.get_logger().info("Robot STOPPED")

        # ── STATUS ───────────────────────────────────────────
        elif cmd == "STATUS":
            state = "LOCKED" if self.is_locked else "UNLOCKED"
            response.success = True
            response.message = (
                f"State={state} | "
                f"vx={self.current_vel.linear.x:.2f} "
                f"vy={self.current_vel.linear.y:.2f} "
                f"wz={self.current_vel.angular.z:.2f}"
            )
            self.get_logger().info(f"Status: {response.message}")

        # ── UNKNOWN ──────────────────────────────────────────
        else:
            response.success = False
            response.message = (
                f"Unknown: '{cmd}'. "
                f"Valid: LOCK | UNLOCK | MOVE | STOP | STATUS"
            )

        response.is_locked         = self.is_locked
        response.current_linear_x  = self.current_vel.linear.x
        response.current_linear_y  = self.current_vel.linear.y
        response.current_angular_z = self.current_vel.angular.z
        return response

    def timed_stop(self):
        self.publish_stop()
        self.get_logger().info("Timed move complete — stopped")


def main(args=None):
    rclpy.init(args=args)
    node = MotionServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()