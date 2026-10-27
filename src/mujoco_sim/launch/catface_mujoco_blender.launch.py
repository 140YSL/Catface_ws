from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 固定参数：按需直接改这里
    blender_file = PathJoinSubstitution([
        FindPackageShare('robot_description'),   # 找到包的 share 目录
        'blender',
        'catcface_skin_fur_v2.blend',
    ])
    blender_receiver = PathJoinSubstitution([
        FindPackageShare('mujoco_sim'),
        'launch',
        'blender_joint_receiver.py',
    ])
    blender_executable = "blender"

    return LaunchDescription([
        # # 启动 Blender GUI（独立进程）
        # ExecuteProcess(
        #     cmd=[blender_executable, blender_file, "--python", blender_receiver],
        #     output="screen",
        #     name="catface_blender",
        # ),

        # # JointState → Blender 的 UDP 桥接
        # Node(
        #     package="mujoco_sim",
        #     executable="mujoco_blender_bridge",
        #     name="mujoco_blender_bridge",
        #     output="log",
        #     parameters=[{
        #         "joint_states_topic": "/joint_states",
        #         "udp_host": "127.0.0.1",
        #         "udp_port": 5005,
        #         "send_rate_hz": 60.0,
        #     }],
        # ),

        # 加载机器人到 MuJoCo
        Node(
            package="mujoco_sim",
            executable="load_robot2mujoco",
            name="load_robot2mujoco",
            output="log",
        ),
    ])