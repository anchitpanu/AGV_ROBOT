from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # ── 1. Camera ─────────────────────────────────────────────
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'video_device':  '/dev/video0',
            'image_width':   640,
            'image_height':  480,
        }],
        remappings=[('/image_raw', '/camera/image_raw')]
    )

    # ── 2. LiDAR ──────────────────────────────────────────────
    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        name='lidar_node',
        output='screen',
        parameters=[{
            'serial_port':      '/dev/ttyUSB0',
            'serial_baudrate':  115200,
            'frame_id':         'laser',
            'angle_compensate': True,
        }]
    )

    # ── 3. Hand gesture → publishes /cmd_vel_raw ──────────────
    hand_node = Node(
        package='agv_control',
        executable='dual_hand_gesture_control',
        name='hand_gesture_node',
        output='screen'
    )

    # ── 4. Motion service → LOCK/UNLOCK, publishes /motion_locked
    service_node = Node(
        package='agv_control',
        executable='motion_service_node',
        name='motion_service_node',
        output='screen'
    )

    # ── 5. Obstacle avoid → /cmd_vel_raw + /scan → /cmd_vel ───
    obstacle_node = Node(
        package='agv_control',
        executable='obstacle_avoid_node',
        name='obstacle_avoid_node',
        output='screen',
        parameters=[{
            'warn_mm':          150.0,
            'crit_mm':          120.0,
            'target_mm':        125.0,
            'escape_lin_speed': 0.12,
            'escape_ang_speed': 0.6,
            'lidar_offset_deg': 0.0,
        }]
    )

    return LaunchDescription([
        camera_node,
        lidar_node,
        service_node,    # start service first so lock status is ready
        hand_node,
        obstacle_node,
    ])