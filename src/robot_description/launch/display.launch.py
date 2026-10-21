import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # 获取 robot_description 功能包的安装路径
    pkg_share = get_package_share_directory('robot_description')

    # URDF 文件路径
    urdf_file = os.path.join(
        pkg_share,
        'urdf',
        'robotic_cathead_v2.urdf'
    )

    # RViz 配置文件路径
    rviz_config_file = os.path.join(
        pkg_share,
        'rviz',
        'urdf.rviz'
    )

    # 读取 URDF 文件
    with open(urdf_file, 'r', encoding='utf-8') as f:
        robot_description_content = f.read()

    return LaunchDescription([

        # 1. 启动 joint_state_publisher_gui
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # 2. 启动 robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_content
            }],
            output='screen'
        ),

        # 3. 启动 RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),

    ])