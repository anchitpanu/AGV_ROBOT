from setuptools import find_packages, setup

package_name = 'agv_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/agv_control/launch', ['launch/bringup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='earn',
    maintainer_email='anchitpan_p@cmu.ac.th',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dual_hand_gesture_control = agv_control.dual_hand_gesture_control:main',
            'obstacle_avoid_node = agv_control.obstacle_avoid_node:main',
            'motion_service_node = agv_control.motion_service_node:main',
        ],
    },
)
