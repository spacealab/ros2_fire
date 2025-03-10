from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
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