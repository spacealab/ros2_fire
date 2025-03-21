from setuptools import setup

package_name = 'fire_detection_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/fire_detection.launch.py']), # اضافه کردن فایل launch
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali Alipour', # اسم خودتون رو بزارید
    maintainer_email='a.alipour@ostfalia.de', # ایمیل خودتون رو بزارید
    description='ROS2 package for fire detection using YOLO', # توضیحات بسته
    license='TODO: License declaration', # لایسنس رو مشخص کنید
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_fire = fire_detection_ros2.webcam_fire:main', # فایل اجرایی
        ],
    },
)