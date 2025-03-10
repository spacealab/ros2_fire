import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ali/Desktop/DigitalTechnologies/KURS/Fire/ROS_PACK/src/install/fire_detection_ros2'
