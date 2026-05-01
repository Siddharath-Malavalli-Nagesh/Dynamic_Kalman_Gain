"""
Launch turtlebot3 in Gazebo + the NCLT-style KalmanNet inference node.

Usage:
  ros2 launch <pkg> gazebo_kalmannet.launch.py weights:=/path/to/best_knet_gazebo.pt
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    weights_arg = DeclareLaunchArgument(
        "weights",
        default_value=os.path.expanduser("~/.ros/best_knet_gazebo.pt"),
        description="Path to fine-tuned KalmanNet weights")

    set_tb3 = SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value="waffle")

    tb3_dir = get_package_share_directory("turtlebot3_gazebo")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_dir, "launch", "empty_world.launch.py")))

    inference = Node(
        package="kalmannet_ros2",
        executable="gazebo_inference_node",
        name="gazebo_inference_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"weights": LaunchConfiguration("weights")}],
    )

    return LaunchDescription([weights_arg, set_tb3, gazebo, inference])
