#!/usr/bin/env python3
"""
dual_hand_gesture_control.py
=============================
Node  : dual_hand_gesture_control
Topics: sub /camera/image_raw  (sensor_msgs/Image)
        pub /cmd_vel           (geometry_msgs/Twist)

Uses mediapipe 0.10.x Tasks API (HandLandmarker).
Model auto-downloaded to ~/hand_landmarker.task on first run.

RIGHT hand  ->  direction
─────────────────────────────────────────────────────────
  ☝  Index only              -> FORWARD
  ✌  Index + Middle          -> BACKWARD
  🖐  Open palm facing cam    -> STRAFE LEFT  (linear.y +)
  🖐  Open palm back to cam   -> STRAFE RIGHT (linear.y -)
  👍 Thumb only              -> ROTATE LEFT  (angular.z +)
  🤙 Thumb + Pinky           -> ROTATE RIGHT (angular.z -)
  ✊  Fist (no thumb)         -> STOP

LEFT hand  ->  speed control
─────────────────────────────────────────────────────────
  Linear  speed : index finger bend angle
                  (straight UP = fast, bent = slow)
  Angular speed : wrist tilt left/right
                  (tilt left = rotate left faster,
                   tilt right = rotate right faster)
  Both shown as bars that FOLLOW the left wrist position.
"""

import os, math, time, urllib.request
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ──────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.expanduser("~/hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand_landmarker.task -> {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")
    return MODEL_PATH

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ──────────────────────────────────────────────────────────────
BUFFER_SIZE       = 7
TIMEOUT_SEC       = 0.7
MIN_ANGLE         = 90.0
MAX_ANGLE         = 175.0
OPEN_PALM_FINGERS = 4

# Wrist tilt thresholds for angular speed (normalised units)
# Tilt = (wrist.x - middle_mcp.x): negative=tilt left, positive=tilt right
TILT_DEAD   = 0.03   # dead-zone  ±3%
TILT_MAX    = 0.15   # full speed ±15%

# Colors (BGR)
C_BG     = ( 35,  35,  40)
C_GRAY   = (180, 180, 180)
C_WHITE  = (255, 255, 255)
C_GREEN  = ( 50, 255,  80)
C_RED    = ( 50,  50, 255)
C_YELLOW = (  0, 255, 255)
C_CYAN   = (255, 230,  50)
C_ORANGE = (  0, 180, 255)
C_PANEL  = ( 55,  55,  62)
C_ACCENT = (  0, 220, 255)

DIR_COLORS = {
    "FORWARD":      C_GREEN,
    "BACKWARD":     C_RED,
    "ROTATE_LEFT":  C_CYAN,
    "ROTATE_RIGHT": C_CYAN,
    "STRAFE_LEFT":  C_YELLOW,
    "STRAFE_RIGHT": C_YELLOW,
    "STOP":         C_RED,
    "NONE":         C_GRAY,
    "TIMEOUT":      C_RED,
}


class DualHandGestureControl(Node):

    def __init__(self):
        super().__init__('dual_hand_gesture_control')

        self.declare_parameter("max_linear_speed",  0.4)
        self.declare_parameter("max_angular_speed", 1.2)
        self.declare_parameter("flip_camera",       True)

        self.max_linear  = self.get_parameter("max_linear_speed").value
        self.max_angular = self.get_parameter("max_angular_speed").value
        self.flip_camera = self.get_parameter("flip_camera").value

        self.bridge = CvBridge()

        # Open camera directly (no need for separate camera node)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera /dev/video0!")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.get_logger().info("Camera opened successfully")

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)

        # Timer to read camera at ~30fps
        self.create_timer(0.033, self.camera_timer_callback)

        model_path = ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Default speed = 20% of max when left hand not visible
        self.default_lin = self.max_linear  * 0.2
        self.default_ang = self.max_angular * 0.2

        # Pre-fill buffers with 20% default
        self.lin_buf = deque([self.default_lin] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
        self.ang_buf = deque([self.default_ang] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
        self.last_seen = time.time()

        # Wrist position for speed-widget follow
        self.wrist_x = 0.25
        self.wrist_y = 0.55

        self.get_logger().info("=== Dual Hand Gesture Control Ready ===")
        self.get_logger().info(
            f"  max_linear={self.max_linear}  max_angular={self.max_angular}"
        )

    # ══════════════════════════════════════════════════════════
    #  Landmark helpers
    # ══════════════════════════════════════════════════════════

    def finger_extended(self, lms, tip, pip):
        return lms[tip].y < lms[pip].y

    def thumb_extended(self, lms):
        """Thumb tip clearly left of IP joint (works after horizontal flip)."""
        return (lms[3].x - lms[4].x) > 0.03

    def is_fist(self, lms):
        """All four fingers folded AND thumb NOT extended."""
        fingers_down = (
            not self.finger_extended(lms,  8,  6) and
            not self.finger_extended(lms, 12, 10) and
            not self.finger_extended(lms, 16, 14) and
            not self.finger_extended(lms, 20, 18)
        )
        return fingers_down and not self.thumb_extended(lms)

    def count_fingers(self, lms):
        tips_pips = [(8,6),(12,10),(16,14),(20,18)]
        return sum(self.finger_extended(lms, t, p) for t, p in tips_pips)

    def palm_facing_camera(self, lms):
        return lms[5].x < lms[17].x

    def hand_size(self, lms):
        w, m = lms[0], lms[9]
        return math.hypot(w.x - m.x, w.y - m.y) + 1e-6

    # ══════════════════════════════════════════════════════════
    #  Direction — RIGHT hand
    # ══════════════════════════════════════════════════════════

    def classify_direction(self, lms):
        """
        Check order (most specific first):
          1. Fist (no thumb)           -> STOP
          2. Thumb + Pinky             -> ROTATE_RIGHT
          3. Thumb only                -> ROTATE_LEFT
          4. Open palm (4+ fingers)    -> STRAFE
          5. Index only                -> FORWARD
          6. Index + Middle            -> BACKWARD
        """
        index  = self.finger_extended(lms,  8,  6)
        middle = self.finger_extended(lms, 12, 10)
        ring   = self.finger_extended(lms, 16, 14)
        pinky  = self.finger_extended(lms, 20, 18)
        thumb  = self.thumb_extended(lms)
        n_up   = self.count_fingers(lms)

        # 1. Fist: all fingers down, no thumb
        if self.is_fist(lms):
            return "STOP"

        # 2. Thumb + Pinky (other fingers down) -> ROTATE RIGHT
        if thumb and pinky and not index and not middle and not ring:
            return "ROTATE_RIGHT"

        # 3. Thumb only (all fingers down) -> ROTATE LEFT
        if thumb and not index and not middle and not ring and not pinky:
            return "ROTATE_LEFT"

        # 4. Open palm (4+ fingers up) -> STRAFE
        if n_up >= OPEN_PALM_FINGERS:
            return "STRAFE_LEFT" if self.palm_facing_camera(lms) else "STRAFE_RIGHT"

        # 5. Index only -> FORWARD
        if index and not middle and not ring and not pinky and not thumb:
            return "FORWARD"

        # 6. Index + Middle -> BACKWARD
        if index and middle and not ring and not pinky:
            return "BACKWARD"

        return "NONE"

    # ══════════════════════════════════════════════════════════
    #  Speed — LEFT hand
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _angle_3pts(a, b, c):
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.hypot(*ba) * math.hypot(*bc)
        if mag < 1e-9:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

    def linear_speed_from_index(self, lms):
        """Index finger straight UP = max speed, bent = 0."""
        angle = self._angle_3pts(lms[5], lms[6], lms[8])
        ratio = (angle - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)
        return max(0.0, min(1.0, ratio)) * self.max_linear

    def angular_speed_from_wrist_tilt(self, lms):
        """
        Wrist tilt = horizontal offset between wrist (0) and middle MCP (9).
        Tilt RIGHT (positive) -> angular speed for right rotation.
        Tilt LEFT  (negative) -> angular speed for left rotation.
        Returns: (angular_speed, tilt_direction)
          tilt_direction: +1 = right, -1 = left, 0 = neutral
        """
        tilt = lms[0].x - lms[9].x   # positive = wrist right of MCP = tilt right

        if abs(tilt) < TILT_DEAD:
            return 0.0, 0

        sign  = 1 if tilt > 0 else -1
        ratio = (abs(tilt) - TILT_DEAD) / (TILT_MAX - TILT_DEAD)
        ratio = max(0.0, min(1.0, ratio))
        return ratio * self.max_angular, sign

    # ══════════════════════════════════════════════════════════
    #  Draw landmarks
    # ══════════════════════════════════════════════════════════

    def draw_landmarks(self, frame, lms, color):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            dot_c = (0, 200, 255) if i == 0 else color
            cv2.circle(frame, pt, 4, dot_c, -1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════
    #  Speed widget (follows left wrist)
    # ══════════════════════════════════════════════════════════

    def draw_speed_widget(self, frame, lin, ang, tilt_dir, has_left=True):
        h, w = frame.shape[:2]
        px = int(self.wrist_x * w)
        py = int(self.wrist_y * h)
        px = max(80, min(w - PANEL_W - 80, px))
        py = max(110, min(h - 30, py))

        bar_h, bar_w = 90, 16
        gap  = 12
        bx1  = px - gap - bar_w   # LIN bar
        bx2  = px + gap            # ANG bar
        by_bot = py - 18
        by_top = by_bot - bar_h

        # Backgrounds
        for bx in [bx1, bx2]:
            cv2.rectangle(frame, (bx, by_top), (bx+bar_w, by_bot), (70,70,75), -1)
            cv2.rectangle(frame, (bx, by_top), (bx+bar_w, by_bot), (140,140,145), 1)

        # LIN fill — cyan when active, gray when default
        lin_fill = int((lin / self.max_linear) * bar_h) if self.max_linear > 0 else 0
        lin_color = C_CYAN if has_left else C_GRAY
        if lin_fill > 0:
            cv2.rectangle(frame, (bx1, by_bot - lin_fill),
                          (bx1 + bar_w, by_bot), lin_color, -1)

        # ANG fill + tilt color — colored when active, gray when default
        ang_fill = int((ang / self.max_angular) * bar_h) if self.max_angular > 0 else 0
        if has_left:
            ang_color = C_CYAN if tilt_dir == -1 else C_ORANGE if tilt_dir == 1 else C_GRAY
        else:
            ang_color = C_GRAY
        if ang_fill > 0:
            cv2.rectangle(frame, (bx2, by_bot - ang_fill),
                          (bx2 + bar_w, by_bot), ang_color, -1)

        # Labels
        cv2.putText(frame, "LIN", (bx1-1, by_top-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, lin_color, 1)
        cv2.putText(frame, f"{lin:.2f}", (bx1-4, by_bot+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, lin_color, 1)

        # ANG label with tilt arrow
        tilt_arrow = " L" if tilt_dir == -1 else " R" if tilt_dir == 1 else ""
        cv2.putText(frame, "ANG" + tilt_arrow, (bx2-1, by_top-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, ang_color, 1)
        cv2.putText(frame, f"{ang:.2f}", (bx2-4, by_bot+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, ang_color, 1)

        # Connector lines to wrist dot
        cv2.line(frame, (bx1 + bar_w//2, by_bot), (px, py), C_ACCENT, 1)
        cv2.line(frame, (bx2 + bar_w//2, by_bot), (px, py), C_ACCENT, 1)

        # Tilt indicator arc
        if tilt_dir != 0:
            arc_r = 22
            start_a = -180 if tilt_dir == -1 else 0
            end_a   =    0 if tilt_dir == -1 else 180
            arc_pts = []
            step = 10 * tilt_dir
            for a in range(start_a, end_a + step, step):
                rad = math.radians(a)
                arc_pts.append((px + int(arc_r * math.cos(rad)),
                                py + int(arc_r * math.sin(rad))))
            for i in range(len(arc_pts) - 1):
                cv2.line(frame, arc_pts[i], arc_pts[i+1], ang_color, 2, cv2.LINE_AA)

        # Wrist anchor dot — gray when using default speed
        dot_color = C_ACCENT if has_left else C_GRAY
        cv2.circle(frame, (px, py),  8, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 12, dot_color,  2, cv2.LINE_AA)

        # Label: DEFAULT or hand icon
        if not has_left:
            cv2.putText(frame, "DEFAULT 20%", (px - 38, py + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1)
        else:
            cv2.putText(frame, "LEFT HAND", (px - 30, py + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_ORANGE, 1)

    # ══════════════════════════════════════════════════════════
    #  Right direction panel
    # ══════════════════════════════════════════════════════════

    def draw_direction_panel(self, panel, direction, lin, ang):
        h, pw = panel.shape[:2]
        dir_color = DIR_COLORS.get(direction, C_GRAY)

        # Header
        cv2.rectangle(panel, (0,0), (pw, 30), (45,45,55), -1)
        cv2.putText(panel, "DIRECTION", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_ACCENT, 1)
        cv2.line(panel, (0,30), (pw,30), (90,90,100), 1)

        # Direction label
        label = direction.replace("_", " ")
        fs = 0.60 if len(label) > 9 else 0.68
        cv2.putText(panel, label, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, dir_color, 2)

        # Robot top-down diagram
        cx, cy = pw // 2, h // 2 - 10
        rw, rh = 28, 38
        # Wheels
        for wx, wy in [(-rw//2-9,-10),(-rw//2-9, 8),(rw//2-1,-10),(rw//2-1, 8)]:
            cv2.rectangle(panel,(cx+wx,cy+wy),(cx+wx+9,cy+wy+14),(110,110,115),-1)
        # Body
        cv2.rectangle(panel,(cx-rw//2,cy-rh//2),(cx+rw//2,cy+rh//2),(65,65,72),-1)
        cv2.rectangle(panel,(cx-rw//2,cy-rh//2),(cx+rw//2,cy+rh//2),(120,120,130), 2)
        # Front marker
        cv2.rectangle(panel,(cx-rw//4,cy-rh//2-7),(cx+rw//4,cy-rh//2),dir_color,-1)
        cv2.putText(panel,"F",(cx-4,cy-rh//2-1),cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,0,0),1)

        # Motion arrow / rotate arc
        alen = 44
        arrows = {
            "FORWARD":      (cx, cy, cx,       cy-alen),
            "BACKWARD":     (cx, cy, cx,       cy+alen),
            "STRAFE_LEFT":  (cx, cy, cx-alen,  cy),
            "STRAFE_RIGHT": (cx, cy, cx+alen,  cy),
        }
        if direction in arrows:
            x1,y1,x2,y2 = arrows[direction]
            cv2.arrowedLine(panel,(x1,y1),(x2,y2),dir_color,3,
                            tipLength=0.30,line_type=cv2.LINE_AA)
        elif direction in ("ROTATE_LEFT","ROTATE_RIGHT"):
            cw = direction == "ROTATE_RIGHT"
            r  = 46
            pts_arc = []
            start_a, end_a, step = (-150,-30,8) if cw else (-30,-150,-8)
            for a in range(start_a, end_a + step, step):
                rad = math.radians(a)
                pts_arc.append((int(cx+r*math.cos(rad)), int(cy+r*math.sin(rad))))
            for i in range(len(pts_arc)-1):
                cv2.line(panel, pts_arc[i], pts_arc[i+1], dir_color, 3, cv2.LINE_AA)
            if len(pts_arc) >= 2:
                cv2.arrowedLine(panel, pts_arc[-2], pts_arc[-1],
                                dir_color, 3, tipLength=1.0)

        # Gesture guide
        cv2.line(panel,(0,h-168),(pw,h-168),(90,90,100),1)
        guide = [
            ("1 finger",     "FORWARD",       "FORWARD"),
            ("2 fingers",    "BACKWARD",      "BACKWARD"),
            ("Thumb",        "ROT LEFT",      "ROTATE_LEFT"),
            ("Thumb+Pinky",  "ROT RIGHT",     "ROTATE_RIGHT"),
            ("Open palm",    "STRAFE",        "STRAFE_LEFT"),
            ("Fist",         "STOP",          "STOP"),
        ]
        gy = h - 156
        for gesture, action, key in guide:
            active = (key == direction) or \
                     (key == "STRAFE_LEFT" and "STRAFE" in direction)
            gc = dir_color if active else (140,140,148)
            prefix = ">" if active else " "
            cv2.putText(panel, f"{prefix} {gesture}", (8, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, gc, 1)
            cv2.putText(panel, action, (120, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, gc, 1)
            gy += 22

        # Speed readout
        cv2.line(panel,(0,h-42),(pw,h-42),(90,90,100),1)
        cv2.putText(panel, f"LIN: {lin:.3f} m/s", (8, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_CYAN, 1)
        cv2.putText(panel, f"ANG: {ang:.3f} r/s", (8, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_ORANGE, 1)

    # ══════════════════════════════════════════════════════════
    #  Main callback
    # ══════════════════════════════════════════════════════════

    def camera_timer_callback(self):
        """Grab frame directly from OpenCV camera."""
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.process_frame(frame)

    def image_callback(self, msg):
        """Fallback: receive frame from ROS topic."""
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_frame(frame)

    def process_frame(self, frame):
        if self.flip_camera:
            frame = cv2.flip(frame, 1)

        # Brighten camera image
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.detector.detect(mp_image)

        right_lms = None
        left_lms  = None

        if result.hand_landmarks:
            for i, lms in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                if label == "Left":             # after flip = user's right
                    right_lms = lms
                    self.draw_landmarks(frame, lms, C_GREEN)
                else:                           # user's left
                    left_lms = lms
                    self.draw_landmarks(frame, lms, C_ORANGE)
                    self.wrist_x = lms[0].x
                    self.wrist_y = lms[0].y
            self.last_seen = time.time()

        # Timeout safety stop
        if time.time() - self.last_seen > TIMEOUT_SEC:
            self.cmd_pub.publish(Twist())
            self._show(frame, "TIMEOUT", 0.0, 0.0, 0, False, False)
            return

        # ── Direction ────────────────────────────────────────
        direction = self.classify_direction(right_lms) if right_lms else "NONE"

        # ── Speed ────────────────────────────────────────────
        tilt_dir = 0
        if left_lms:
            # Left hand visible: map full range 0 -> max
            lin = self.linear_speed_from_index(left_lms)
            ang, tilt_dir = self.angular_speed_from_wrist_tilt(left_lms)
            self.lin_buf.append(lin)
            self.ang_buf.append(ang)
        else:
            # No left hand: hold buffer at 50% default (don't push 0)
            # Only refill if buffer has drifted away from default
            current_avg = sum(self.lin_buf) / len(self.lin_buf)
            if abs(current_avg - self.default_lin) > 0.01:
                self.lin_buf.append(self.default_lin)
                self.ang_buf.append(self.default_ang)

        lin_spd = sum(self.lin_buf) / len(self.lin_buf)
        ang_spd = sum(self.ang_buf) / len(self.ang_buf)

        # ── Twist ────────────────────────────────────────────
        twist = Twist()
        if   direction == "FORWARD":
            twist.linear.x  =  lin_spd
        elif direction == "BACKWARD":
            twist.linear.x  = -lin_spd
        elif direction == "STRAFE_LEFT":
            twist.linear.y  =  lin_spd
        elif direction == "STRAFE_RIGHT":
            twist.linear.y  = -lin_spd
        elif direction == "ROTATE_LEFT":
            twist.angular.z =  ang_spd
        elif direction == "ROTATE_RIGHT":
            twist.angular.z = -ang_spd

        self.cmd_pub.publish(twist)
        self._show(frame, direction, lin_spd, ang_spd, tilt_dir,
                   right_lms is not None, left_lms is not None)

    # ══════════════════════════════════════════════════════════
    #  Build & show display
    # ══════════════════════════════════════════════════════════

    PANEL_W = 210

    def _show(self, frame, direction, lin, ang, tilt_dir, has_right, has_left):
        h, w = frame.shape[:2]

        # Speed widget on camera frame (always shown, grayed out when no left hand)
        self.draw_speed_widget(frame, lin, ang, tilt_dir, has_left)

        # Hand label hints
        if has_right:
            cv2.putText(frame, "RIGHT: direction", (w-175, h-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREEN, 1)
        if has_left:
            cv2.putText(frame, "LEFT: speed", (w-175, h-36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ORANGE, 1)

        # Top status bar
        cv2.rectangle(frame, (0,0), (w, 28), (40,40,48), -1)
        cv2.putText(frame, "Dual Hand Gesture Control",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_ACCENT, 1)
        r_c = C_GREEN  if has_right else C_GRAY
        l_c = C_ORANGE if has_left  else C_GRAY
        cv2.circle(frame, (w-55, 14), 5, r_c, -1)
        cv2.putText(frame, "R", (w-47, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, r_c, 1)
        cv2.circle(frame, (w-28, 14), 5, l_c, -1)
        cv2.putText(frame, "L", (w-20, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, l_c, 1)

        # Direction panel
        panel = np.full((h, self.PANEL_W, 3), (28,28,34), dtype=np.uint8)
        cv2.line(panel, (0,0), (0,h), (200,200,210), 1)
        self.draw_direction_panel(panel, direction, lin, ang)

        combined = np.hstack([frame, panel])
        cv2.imshow("Dual Hand Gesture Control", combined)
        cv2.waitKey(1)


PANEL_W = 210   # module-level so draw_speed_widget can offset correctly


def main(args=None):
    rclpy.init(args=args)
    node = DualHandGestureControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'cap') and node.cap.isOpened():
            node.cap.release()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
