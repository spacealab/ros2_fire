from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # نود وب‌کم
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            parameters=[
                {'video_device': '/dev/video0'},
                {'image_width': 640},
                {'image_height': 480},
                {'camera_name': 'webcam'},
                {'pixel_format': 'yuyv'},
            ],
            remappings=[
                ('/image_raw', '/image_raw'),
            ]
        ),
        # نود webcam_fire
        Node(
            package='fire_detection_ros2',
            executable='webcam_fire',
            name='webcam_fire',
            namespace='fire_detection',
            output='screen',
            remappings=[
                ('image_raw', '/image_raw'),
            ],
        ),
    ])