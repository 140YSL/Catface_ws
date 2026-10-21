#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MuJoCo ROS 2 node that publishes scalar joint states and consumes
joint commands from a control topic (std_msgs/Float64MultiArray).

新增加功能：
- 订阅控制话题（默认 /mujoco_ctrl，可通过参数 ctrl_topic 修改）
- 消息为长度为 model.nu 的浮点数组，按执行器顺序对应 data.ctrl
- 收到指令后在仿真步进时（持锁）写入 data.ctrl
- 接受指令的14为执行器，分别为：
    0: neck_pitch
    1: neck_yaw
    2: Head
    3: Jaw
    4: left_upper_palate
    5: right_upper_palate
    6: left_eye_bottom_skin
    7: left_eye_upper_skin
    8: right_eye_bottom_skin
    9: right_eye_upper_skin
    10: left_ear_rotate
    11: left_ear
    12: right_ear_rotate
    13: right_ear
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory
import os

XML_PATH = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'robotic_cathead_v2.xml'
    )


class MuJoCoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim_node")

        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("state_publish_rate_hz", 50.0)
        self.declare_parameter("ctrl_topic", "/mujoco_ctrl")
        self.joint_states_topic = str(
            self.get_parameter("joint_states_topic").value
        )
        self.state_publish_rate_hz = max(
            1.0, float(self.get_parameter("state_publish_rate_hz").value)
        )
        self.publish_period = 1.0 / self.state_publish_rate_hz
        self.ctrl_topic = str(self.get_parameter("ctrl_topic").value)

        self.joint_state_publisher = self.create_publisher(
            JointState, self.joint_states_topic, 10
        )

        self.model = None
        self.data = None
        self.viewer = None
        self.simulation_timer = None
        self.joint_state_timer = None
        self.scalar_joint_addresses = []

        # 最新收到的控制指令（长度为 model.nu 的数组）
        self._latest_ctrl = None  # type: np.ndarray | None
        self.ctrl_subscription = None

        self.get_logger().info("MuJoCo 仿真节点启动")
        try:
            self.load_robot()
            self.configure_joint_state_publisher()
            self.configure_ctrl_subscription()
        except Exception:
            self.cleanup()
            raise

        self.simulation_timer = self.create_timer(
            max(float(self.model.opt.timestep), 0.0001),
            self.simulation_callback,
        )
        self.joint_state_timer = self.create_timer(
            self.publish_period, self.publish_joint_states
        )

        self.get_logger().info("机器人模型加载成功")
        self.get_logger().info(f"刚体数量: {self.model.nbody}")
        self.get_logger().info(f"MuJoCo 关节数量: {self.model.njnt}")
        self.get_logger().info(f"广义坐标数量 nq: {self.model.nq}")
        self.get_logger().info(f"速度自由度数量 nv: {self.model.nv}")
        self.get_logger().info(f"执行器数量: {self.model.nu}")
        self.get_logger().info(f"JointState 话题: {self.joint_states_topic}")
        self.get_logger().info(
            f"JointState 发布频率: {self.state_publish_rate_hz:.1f} Hz"
        )
        self.get_logger().info(f"控制指令话题: {self.ctrl_topic} (Float64MultiArray, 长度={self.model.nu})")

    def load_robot(self):
        if not os.path.isfile(XML_PATH):
            raise FileNotFoundError(f"找不到 MJCF 文件: {XML_PATH}")

        self.get_logger().info(f"正在加载 MJCF: {XML_PATH}")
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def configure_joint_state_publisher(self):
        """Publish scalar hinge/slide joints; skip free and ball joints."""
        self.scalar_joint_addresses.clear()
        skipped = []
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)

        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not joint_name:
                joint_name = f"joint_{joint_id}"
                self.get_logger().warning(
                    f"MuJoCo joint {joint_id} 没有名称，临时命名为 "
                    f"{joint_name}，可能无法与 URDF 匹配。"
                )

            joint_type = int(self.model.jnt_type[joint_id])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            qvel_address = int(self.model.jnt_dofadr[joint_id])

            if joint_type in (hinge_type, slide_type):
                self.scalar_joint_addresses.append(
                    (joint_name, qpos_address, qvel_address)
                )
            else:
                skipped.append(joint_name)

        if skipped:
            self.get_logger().warning(
                "以下非标量关节（例如 FREE/BALL）不会发布到 JointState: "
                + ", ".join(skipped)
            )

        names = [entry[0] for entry in self.scalar_joint_addresses]
        self.get_logger().info(
            f"可发布的标量关节数量: {len(names)}"
            + (f"；关节名: {', '.join(names)}" if names else "")
        )
        if not names:
            self.get_logger().warning(
                "没有找到 hinge/slide 关节，请检查 MJCF 中的关节定义。"
            )

    def configure_ctrl_subscription(self):
        """订阅控制话题：Float64MultiArray，长度必须等于执行器数量。"""
        if self.model.nu <= 0:
            self.get_logger().warning("模型中没有执行器，忽略控制话题订阅。")
            return
        self.ctrl_subscription = self.create_subscription(
            Float64MultiArray,
            self.ctrl_topic,
            self.ctrl_callback,
            10,
        )

    def ctrl_callback(self, msg: Float64MultiArray):
        """缓存最新控制指令，由仿真线程在持锁状态下写入 data.ctrl。"""
        if self.model is None:
            return
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size != self.model.nu:
            self.get_logger().warning(
                f"控制指令长度 {values.size} 与执行器数量 {self.model.nu} "
                "不一致，已忽略本条指令。"
            )
            return
        self._latest_ctrl = values.copy()

    def simulation_callback(self):
        if self.viewer is None or not self.viewer.is_running():
            self.get_logger().info("MuJoCo 查看器已关闭，停止节点")
            if rclpy.ok():
                rclpy.shutdown()
            return

        # Passive viewer: the application is responsible for stepping.
        with self.viewer.lock():
            if self._latest_ctrl is not None:
                self.data.ctrl[:] = self._latest_ctrl
            mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def publish_joint_states(self):
        if self.model is None or self.data is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # The passive viewer has its own thread. Use its lock while reading
        # simulation state so this callback cannot read arrays mid-update.
        if self.viewer is not None:
            with self.viewer.lock():
                for joint_name, qpos_address, qvel_address in self.scalar_joint_addresses:
                    msg.name.append(joint_name)
                    msg.position.append(float(self.data.qpos[qpos_address]))
                    msg.velocity.append(float(self.data.qvel[qvel_address]))
        else:
            for joint_name, qpos_address, qvel_address in self.scalar_joint_addresses:
                msg.name.append(joint_name)
                msg.position.append(float(self.data.qpos[qpos_address]))
                msg.velocity.append(float(self.data.qvel[qvel_address]))

        # Effort is omitted because this node does not calculate actuator torque
        # per joint for the JointState message.
        if msg.name:
            self.joint_state_publisher.publish(msg)

    def cleanup(self):
        for timer_attr in ("simulation_timer", "joint_state_timer"):
            timer = getattr(self, timer_attr, None)
            if timer is not None:
                try:
                    self.destroy_timer(timer)
                except Exception:
                    pass
                setattr(self, timer_attr, None)

        if self.ctrl_subscription is not None:
            try:
                self.destroy_subscription(self.ctrl_subscription)
            except Exception:
                pass
            self.ctrl_subscription = None

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MuJoCoSimNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"MuJoCo 仿真启动失败: {exc}")
        raise
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
