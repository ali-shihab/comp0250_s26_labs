from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cw2_world_spawner',
            executable='world_spawner.py',
            name='cw2_world_spawner',
            output='screen',
        )
    ])
