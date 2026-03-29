from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_delay = LaunchConfiguration('launch_delay')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gazebo_gui = LaunchConfiguration('use_gazebo_gui')
    use_rviz = LaunchConfiguration('use_rviz')
    enable_realsense = LaunchConfiguration('enable_realsense')
    enable_camera_processing = LaunchConfiguration('enable_camera_processing')
    camera_processing_out_cloud = LaunchConfiguration('camera_processing_out_cloud')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    pointcloud_qos_reliable = LaunchConfiguration('pointcloud_qos_reliable')
    use_software_rendering = LaunchConfiguration('use_software_rendering')

    return LaunchDescription([
        DeclareLaunchArgument('launch_delay', default_value='5.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_gazebo_gui', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('enable_realsense', default_value='true'),
        DeclareLaunchArgument('enable_camera_processing', default_value='false'),
        DeclareLaunchArgument(
            'camera_processing_out_cloud',
            default_value='/r200/camera/depth_registered/points'),
        DeclareLaunchArgument(
            'pointcloud_topic',
            default_value='/r200/camera/depth_registered/points'),
        DeclareLaunchArgument('pointcloud_qos_reliable', default_value='true'),
        DeclareLaunchArgument('use_software_rendering', default_value='false'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('rpl_panda_with_rs'),
                '/launch/display.launch.py'
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'use_gazebo_gui': use_gazebo_gui,
                'use_rviz': use_rviz,
                'enable_realsense': enable_realsense,
                'enable_camera_processing': enable_camera_processing,
                'camera_processing_out_cloud': camera_processing_out_cloud,
                'use_software_rendering': use_software_rendering,
                'extra_gazebo_model_path': PathJoinSubstitution([
                    FindPackageShare('cw2_world_spawner'),
                    'models',
                ]),
            }.items()
        ),

        TimerAction(
            period=launch_delay,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        FindPackageShare('cw2_world_spawner'),
                        '/launch/world_spawner.launch.py'
                    ])
                )
            ]
        ),

        Node(
            package='cw2_team_x',
            executable='cw2_solution_node',
            name='cw2_solution_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'pointcloud_topic': pointcloud_topic,
                'pointcloud_qos_reliable': pointcloud_qos_reliable,
            }],
        ),
    ])
