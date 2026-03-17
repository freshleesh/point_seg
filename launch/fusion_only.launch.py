import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("point_seg")
    config_file = os.path.join(pkg_share, "config", "point_seg.yaml")

    return LaunchDescription(
        [
            Node(
                package="point_seg",
                executable="fusion_seg_paint_node",
                name="point_seg_node",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
