#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""键盘遥测节点：按键控制各关节目标角度缓慢变化，并发布到 /mujoco_ctrl。

按键映射（按住生效，松开停止变化）：
    z: neck_pitch              x: neck_yaw              c: Head
    v: Jaw                     q: left_upper_palate     a: right_upper_palate
    w: left_eye_bottom_skin    e: left_eye_upper_skin   s: right_eye_bottom_skin
    d: right_eye_upper_skin    r: left_ear_rotate       t: left_ear
    f: right_ear_rotate        g: right_ear

    p: 切换角度变化方向（正/反）

注意：需要在普通终端中直接运行（ros2 run 或 python3 均可），不支持 ssh 之外的
非 TTY 环境；按 Ctrl+C 退出并恢复终端设置。
"""

import select
import sys
import termios
import tty
from pathlib import Path
import os

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory

# ==================== 宏参数区（可按需修改） ====================

# MJCF 模型路径，用于解析执行器索引
XML_PATH = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'robotic_cathead_v2.xml'
    )

# 发布话题与频率
CTRL_TOPIC = "/mujoco_ctrl"
PUBLISH_RATE_HZ = 50.0

# 角度变化速度（rad/s）：按住某键时，对应关节目标角度每秒变化量
ANGLE_VELOCITY = 0.3

# 目标角度限幅（rad），防止按键过久导致角度发散
ANGLE_LIMIT = 1.0

# 按键 → 执行器名 映射表
KEY_ACTUATOR_MAP = {
    "z": "neck_pitch",
    "x": "neck_yaw",
    "c": "Head",
    "v": "Jaw",
    "q": "left_upper_palate",
    "a": "right_upper_palate",
    "w": "left_eye_bottom_skin",
    "e": "left_eye_upper_skin",
    "s": "right_eye_bottom_skin",
    "d": "right_eye_upper_skin",
    "r": "left_ear_rotate",
    "t": "left_ear",
    "f": "right_ear_rotate",
    "g": "right_ear",
}

# 反向切换按键
DIRECTION_TOGGLE_KEY = "p"

# =============================================================


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("keyboard_teleop_node")

        self.publish_rate_hz = max(1.0, float(PUBLISH_RATE_HZ))
        self.angle_velocity = float(ANGLE_VELOCITY)
        self.angle_limit = abs(float(ANGLE_LIMIT))

        self.get_logger().info("正在加载 MJCF 以解析执行器索引 ...")
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.nu = int(self.model.nu)
        self.get_logger().info(f"执行器数量: {self.nu}")

        # 预解析 按键字符 → 执行器索引，跳过未找到的执行器
        self.key_to_aidx = {}
        for key, name in KEY_ACTUATOR_MAP.items():
            idx = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if idx < 0:
                self.get_logger().warning(
                    f"未找到执行器 '{name}'（按键 '{key}'），已跳过。"
                )
                continue
            self.key_to_aidx[key] = idx

        # 当前被按下的按键集合
        self.pressed_keys = set()
        # 角度变化方向：+1 正向 / -1 反向，按 p 切换
        self.direction = 1.0

        # 当前指令（与演示节点中 current_ctrl 的角色相同）
        self.current_ctrl = np.zeros(self.nu)

        self.publisher = self.create_publisher(
            Float64MultiArray, CTRL_TOPIC, 10
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz, self.tick
        )

        # 进入 cbreak 模式，按键即时可读且无需回车
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

        self.get_logger().info(
            f"键盘遥测节点启动，发布到 {CTRL_TOPIC} @ {self.publish_rate_hz:.1f} Hz，"
            f"速度 {self.angle_velocity} rad/s，限幅 ±{self.angle_limit} rad"
        )
        self.get_logger().info(
            "按键映射: " + ", ".join(
                f"{k}={v}" for k, v in KEY_ACTUATOR_MAP.items()
            ) + f"；按 {DIRECTION_TOGGLE_KEY} 切换方向，Ctrl+C 退出"
        )

    # ---------------- 键盘读取 ----------------

    def poll_keyboard(self):
        """非阻塞读取当前可用的按键事件，更新 pressed_keys 集合。"""
        while True:
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
            except (OSError, ValueError):
                return
            if not rlist:
                return
            ch = sys.stdin.read(1)
            if not ch:
                return
            # 忽略方向键等转义序列的后续字节
            if ch == "\x1b":
                continue
            if ch in self.key_to_aidx:
                self.pressed_keys.add(ch)
            elif ch == DIRECTION_TOGGLE_KEY:
                self.direction *= -1.0
                self.get_logger().info(
                    f"方向切换：{'正向 +' if self.direction > 0 else '反向 -'}"
                )
            elif ch in ("\x03",):  # Ctrl+C
                raise KeyboardInterrupt
            # 松开检测：cbreak 模式下普通按键不产生释放事件，
            # 因此采用“边读边集、周期清空”的方式近似：
            # 这里记录本次 tick 内读到的按键，交由 tick 末尾处理。

    # ---------------- 主循环 ----------------

    def tick(self):
        # 每个周期先清空，再读入本周期内按住的键；
        # 持续按住时终端会重复发送该键，从而保持集合非空。
        self.pressed_keys.clear()
        self.poll_keyboard()

        dt = 1.0 / self.publish_rate_hz
        delta = self.direction * self.angle_velocity * dt

        for key in self.pressed_keys:
            idx = self.key_to_aidx[key]
            self.current_ctrl[idx] = np.clip(
                self.current_ctrl[idx] + delta,
                -self.angle_limit,
                self.angle_limit,
            )

        msg = Float64MultiArray()
        msg.data = self.current_ctrl.tolist()
        self.publisher.publish(msg)

    # ---------------- 退出清理 ----------------

    def restore_terminal(self):
        try:
            termios.tcsetattr(
                self._fd, termios.TCSADRAIN, self._old_settings
            )
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.restore_terminal()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()