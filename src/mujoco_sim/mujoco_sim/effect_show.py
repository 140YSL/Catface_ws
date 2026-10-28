#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运动控制节点：按三组正弦动作生成 ctrl 指令并发布到 /mujoco_ctrl。

与独立脚本 mujoco_motion_demo.py 的控制逻辑保持一致：
- 动作1: 耳朵 + 颅耳 + 下巴 (2.0 rad/s)
- 动作2: 眼部皮肤（top 反相 → 眨眼）(3.0 rad/s)
- 动作3: 颈部俯仰 (1.0 rad/s)
- 每组持续 5s，组间 0.5s 淡入淡出，循环播放
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
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

XML_PATH = Path(
    "/home/ysl/Desktop/CatFace/Catface_ws/src/"
    "robot_description/urdf/robotic_cathead_v2.xml"
)


class MuJoCoMotionCommandNode(Node):
    def __init__(self):
        super().__init__("mujoco_motion_command_node")

        self.declare_parameter("ctrl_topic", "/mujoco_ctrl")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.ctrl_topic = str(self.get_parameter("ctrl_topic").value)
        self.publish_rate_hz = max(
            1.0, float(self.get_parameter("publish_rate_hz").value)
        )

        self.get_logger().info("正在加载 MJCF 以解析执行器索引 ...")
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.nu = int(self.model.nu)
        self.get_logger().info(f"执行器数量: {self.nu}")

        # ============ 三种运动组定义 ============
        # 格式: "actuator名": (幅值, 频率Hz)
        groups = [
            {   # 动作1: 耳朵 + 颅耳 + 下巴
                "Jaw":            (0.4, 0.4),
                "left_ear":       (0.3, 0.4),
                "right_ear":      (-0.3, 0.4),
                "left_ear_rotate": (0.3, 0.4),
                "right_ear_rotate":(-0.3, 0.4),
            },
            {   # 动作1: 耳朵 + 颅耳 + 下巴
                "left_upper_palate": (0.3, 0.4),
                "right_upper_palate":(-0.3, 0.4),
            },
            {   # 动作2: 眼部皮肤（左右对称，上下反相 → 眨眼）
                "left_eye_bottom_skin":  (-1.0, 0.4),
                "right_eye_bottom_skin": (1.0, 0.4),
                "left_eye_upper_skin":     (1.0, 0.4),
                "right_eye_upper_skin":    (-1.0, 0.4),
            },
            {   # 动作3: 颈部
                "neck_pitch": (0.5, 0.2),
            },
            {   # 动作3: 颈部
                "neck_yaw": (0.5, 0.2),
            },
            {   # 动作3: 颈部
                "Head": (0.5, 0.2),
            },
        ]

        def aid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # 预解析索引，并记录每个执行器 id → 名称（用于 top 反相判断）
        self.parsed_groups = []
        self.actuator_names = {}
        for g in groups:
            parsed = {}
            for name, (amp, omega) in g.items():
                idx = aid(name)
                if idx < 0:
                    self.get_logger().warning(f"未找到执行器 '{name}'，已跳过。")
                    continue
                parsed[idx] = (amp, omega)
                self.actuator_names[idx] = name
            self.parsed_groups.append(parsed)

        self.PHASE_DURATION = 5.0   # 每个动作持续 5 秒

        # 当前指令（用于淡入淡出插值的初值），与独立脚本中 data.ctrl 的角色相同
        self.current_ctrl = np.zeros(self.nu)

        self.publisher = self.create_publisher(
            Float64MultiArray, self.ctrl_topic, 10
        )
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.tick)

        self.get_logger().info(
            f"控制节点启动，发布到 {self.ctrl_topic} @ {self.publish_rate_hz:.1f} Hz"
        )

    def compute_command(self, t: float) -> np.ndarray:
        total = len(self.parsed_groups) * self.PHASE_DURATION
        t_cycle = t % total
        phase = int(t_cycle // self.PHASE_DURATION)
        t_in_phase = t_cycle - phase * self.PHASE_DURATION

        ctrl = np.zeros(self.nu)
        for idx, (amp, omega_hz) in self.parsed_groups[phase].items():
            # 眼部 top 反相，实现上下合拢的眨眼效果
            # sign = -1.0 if "top" in self.actuator_names[idx] else 1.0
            ctrl[idx] = amp * np.sin(omega_hz * 2 * np.pi * t_in_phase)
        return ctrl

    def tick(self):
        t = self.get_clock().now().nanoseconds / 1e9
        self.current_ctrl = self.compute_command(t)

        msg = Float64MultiArray()
        msg.data = self.current_ctrl.tolist()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MuJoCoMotionCommandNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
