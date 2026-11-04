from ament_index_python.packages import get_package_share_directory

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from scipy.interpolate import interp1d

_SHARE_DIR = get_package_share_directory("au_player")
TIME2AU_PATH = f"{_SHARE_DIR}/config/39_smooth.npz"
AU2JOINT_INDEX_PATH = f"{_SHARE_DIR}/config/au2joint_smooth.npz"
AU2JOINT_DATA_PATH = f"{_SHARE_DIR}/config/au2joint_smooth.npy"

"""
    /mujoco_ctrl 各位含义
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


AU_NAMES = list(np.load(AU2JOINT_INDEX_PATH, allow_pickle=True)["au_names"])
JOINT_NAMES = list(np.load(AU2JOINT_INDEX_PATH, allow_pickle=True)["joint_names"])
AU_IDX = np.arange(len(AU_NAMES))
AU_DESCRIPTIONS = {
    "LAU143": "左眼闭眼",
    "RAU143": "右眼闭眼",
    "AU109+110": "皱鼻+上唇提升",
    "AU118": "唇部前撅",
    "AU26": "下颌下垂",
    "LEAD101": "左耳向前",
    "LEAD104": "左耳旋转",
    "LEAD106": "左耳向后",
    "READ101": "右耳向前",
    "READ104": "右耳旋转",
    "READ106": "右耳向后",
}


class AU_player_Node(Node):
    def __init__(self):
        super().__init__('au_player_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/mujoco_ctrl', 10)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz
        self.joint_values = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.srv = self.create_service(Trigger, 'start_au_player', self.start_callback)

        self.time2au_data = load_from_npz(TIME2AU_PATH)
        self.au2joint_data = load_from_npy(AU2JOINT_DATA_PATH)
        self.total_time = float(self.time2au_data["time"][-1])

        self.playing = False
        self.start_time = None

    def start_callback(self, request, response):
        if self.playing:
            response.success = False
            response.message = "AU player is already playing."
            return response
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.playing = True
        response.success = True
        response.message = "AU player started."
        self.get_logger().info("收到 start_au_player 服务调用，开始播放。")
        return response

    def timer_callback(self):
        if self.playing:
            now = self.get_clock().now().nanoseconds / 1e9
            dt = now - self.start_time
            if dt > self.total_time:
                self.playing = False
                self.start_time = None
                self.joint_values = np.zeros(len(JOINT_NAMES), dtype=np.float64)
                self.get_logger().info("播放结束，重置状态，等待下一次服务调用。")
            else:
                self.joint_values = change_time2joint(
                    self.time2au_data, self.au2joint_data, dt)
        msg = Float64MultiArray()
        msg.data = self.joint_values.tolist()
        self.publisher_.publish(msg)


def load_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    return data

def load_from_npy(npy_path):
    data = np.load(npy_path, allow_pickle=False)
    return data

def change_time2joint(time2au_data, au2joint_data, time):
    n_strength = au2joint_data.shape[1]
    au_strength = np.linspace(0, 5, n_strength)

    # 1) 沿时间轴插值：一次算出所有 AU 在 time 处的值
    au_matrix = np.column_stack([time2au_data[name] for name in AU_NAMES])  # (T, N_AU)
    f_time = interp1d(time2au_data["time"], au_matrix, axis=0,
                      assume_sorted=True, bounds_error=False,
                      fill_value=(au_matrix[0], au_matrix[-1]))  # 与 np.interp 一样越界取端点
    au_values = np.atleast_1d(f_time(time)).ravel()             # (N_AU,)

    # 2) 沿 AU 强度轴插值：每个 AU 用自己的强度值（配对），用广播手工线性插值
    idx = np.searchsorted(au_strength, au_values) - 1
    idx = np.clip(idx, 0, n_strength - 2)                       # (N_AU,)，clip 模拟 np.interp 的越界截断
    x0, x1 = au_strength[idx], au_strength[idx + 1]
    y0 = au2joint_data[AU_IDX, idx]                             # (N_AU, N_joint)
    y1 = au2joint_data[AU_IDX, idx + 1]
    w = ((au_values - x0) / (x1 - x0))[:, None]                 # (N_AU, 1)

    joint_values = (y0 + w * (y1 - y0)).sum(axis=0)             # (N_joint,)
    return joint_values

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AU_player_Node()
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

