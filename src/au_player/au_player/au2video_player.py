"""按 AU_DESCRIPTIONS 顺序逐个播放 AU，供视频录制使用。"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from au_player.au_player import (
    AU2JOINT_DATA_PATH,
    AU2JOINT_INDEX_PATH,
    AU_DESCRIPTIONS,
)


def load_playback_points(data_path, index_path):
    """按名称匹配 AU，返回每个 AU 的 0、0.2、……、5 档关节角度。"""
    data = np.load(data_path, allow_pickle=False)
    with np.load(index_path, allow_pickle=False) as index:
        au_names = index['au_names'].tolist()
        joint_names = index['joint_names'].tolist()
        intensities = index['intensities'].copy()

    expected_shape = (len(au_names), len(intensities), len(joint_names))
    if data.shape != expected_shape or not np.isfinite(data).all():
        raise ValueError(f'关节映射数据无效，预期形状为 {expected_shape}')
    if (intensities.ndim != 1 or not np.isfinite(intensities).all()
            or not np.all(np.diff(intensities) > 0)):
        raise ValueError('强度索引必须是一维、有限且严格递增的数组')

    strengths = np.arange(26, dtype=np.float64) / 5.0
    strength_indices = np.abs(
        intensities[:, None] - strengths[None, :]
    ).argmin(axis=0)
    if not np.allclose(intensities[strength_indices], strengths,
                       rtol=0, atol=1e-8):
        raise ValueError('映射表必须包含 0 到 5、步长为 0.2 的强度点')

    # 其余 AU 始终为零；保留原 au_player 对所有 AU 关节贡献求和的规则。
    if not np.isclose(intensities[0], 0.0):
        raise ValueError('映射表的起始强度必须为零')
    neutral = data[:, 0, :].sum(axis=0)
    points = []
    for name in AU_DESCRIPTIONS:
        if au_names.count(name) != 1:
            raise ValueError(f'AU 索引必须恰好包含一个 {name}')
        au_index = au_names.index(name)
        for strength, strength_index in zip(strengths, strength_indices):
            joints = neutral - data[au_index, 0] + data[au_index, strength_index]
            points.append((name, float(strength), joints))
    return points, neutral


class AU2VideoPlayer(Node):
    """自动播放一轮；每个点保持一秒，结束后持续发布零强度姿态。"""

    def __init__(self):
        super().__init__('au2video_player')
        self.points, self.neutral = load_playback_points(
            AU2JOINT_DATA_PATH, AU2JOINT_INDEX_PATH)
        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/mujoco_ctrl', 10)
        self.joint_values = self.neutral.copy()
        self.point_index = 0
        self.last_point_time = None
        self.finished = False
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info('等待 /mujoco_ctrl 订阅者，连接后自动开始播放。')

    def timer_callback(self):
        """以 50 Hz 发布姿态，每隔一秒取下一个强度点。"""
        if self.last_point_time is None:
            if self.publisher_.get_subscription_count() == 0:
                return

        now = time.monotonic()
        if not self.finished and (
                self.last_point_time is None
                or now - self.last_point_time >= 1.5):
            if self.point_index == len(self.points):
                self.joint_values = self.neutral.copy()
                self.finished = True
                self.get_logger().info('11 个 AU 播放完毕，所有 AU 强度已归零。')
            else:
                name, strength, self.joint_values = self.points[self.point_index]
                self.point_index += 1
                self.get_logger().info(
                    f'{name}（{AU_DESCRIPTIONS[name]}）强度 {strength:.1f}')
            self.last_point_time = now

        msg = Float64MultiArray()
        msg.data = self.joint_values.tolist()
        self.publisher_.publish(msg)


def main(args=None):
    """启动自动播放节点。"""
    rclpy.init(args=args)
    node = None
    try:
        node = AU2VideoPlayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
