import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    livox_launch_file = os.path.join(
        get_package_share_directory("livox_ros_driver2"),
        "launch_ROS2",
        "rviz_MID360_launch.py",
    )
    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(livox_launch_file)
    )

    return LaunchDescription(
        [
            Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                name="v4l2_camera",
                output="screen",
            ),
            livox_launch,
        ]
    )
