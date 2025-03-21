import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ali/Desktop/DigitalTechnologies/Fire/ros_files/ROS_PACK/install/fire_detection_ros2'
