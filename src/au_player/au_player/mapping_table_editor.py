#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AU2Joint 编辑器

功能说明:
- 维护一个三维数组 (AU种类 × 强度等级 × 关节角度)
- 通过 GUI 表格选择 AU 和强度
- 从 ROS2 话题 /joint_state_blender 读取关节角度
- 保存到指定的 (AU, 强度) 位置

使用方法:
    python au2joint_editor.py

所有配置参数均在 macros.py 中定义
"""

import sys
import os

import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QMessageBox,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QFont

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ============================================
# 路径配置
# ============================================
# 脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件保存路径 (相对于脚本位置的 ../config/au2joint.npy)
DATA_FILE_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "config", "au2joint.npy"))

# 确保 config 目录存在
CONFIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "config"))

# ============================================
# AU (Action Unit) 配置
# ============================================
# 11 种 AU 名称列表 (横坐标)
AU_NAMES = [
    "LAU143",
    "RAU143",
    "AU109+110",
    "AU118",
    "AU26",
    "LEAD101",
    "LEAD104",
    "LEAD106",
    "READ101",
    "READ104",
    "READ106",
]

# AU 补充说明 (与 AU_NAMES 按顺序一一对应)
AU_DESCRIPTION = [
    "左眼睑下垂",
    "右眼睑下垂",
    "挑眉",
    "嘴唇皱起",
    "下颌下落",
    "左眉内侧上扬",
    "左眉外侧上扬",
    "左眉降低",
    "右眉内侧上扬",
    "右眉外侧上扬",
    "右眉降低",
]

# AU 数量
AU_COUNT = len(AU_NAMES)

# ============================================
# 强度配置
# ============================================
# 强度等级 1-5 (纵坐标)
INTENSITY_LEVELS = [1, 2, 3, 4, 5]

# 强度数量
INTENSITY_COUNT = len(INTENSITY_LEVELS)

# ============================================
# 关节配置
# ============================================
# 关节角度列表长度
JOINT_ANGLE_COUNT = 14

# ============================================
# GUI 配置
# ============================================
# 窗口标题
WINDOW_TITLE = "AU2Joint 编辑器"

# 窗口初始大小
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 700

# ============================================
# 数据初始化
# ============================================
# 初始化填充值 (未设置时的默认值)
DEFAULT_ANGLE_VALUE = 0.0


class ROS2ListenerThread(QThread):
    """ROS2 监听线程，用于在后台运行 rclpy.spin()"""

    def __init__(self, node):
        super().__init__()
        self.node = node

    def run(self):
        rclpy.spin(self.node)


class AU2JointEditor(QMainWindow):
    """AU2Joint 编辑器主窗口"""

    # 信号：收到新的 JointState 消息
    joint_state_received = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # 当前选中的 (AU索引, 强度索引)
        self.selected_au_idx = -1
        self.selected_intensity_idx = -1

        # 最新的关节状态消息
        self.latest_joint_state = None

        # 三维数据数组 (AU_COUNT × INTENSITY_COUNT × JOINT_ANGLE_COUNT)
        self.data = None

        # ROS2 节点和线程
        self.ros_node = None
        self.ros_thread = None

        self.init_ui()
        self.init_data()

        # 连接信号槽（必须在 ROS 初始化之前，确保消息不会丢失）
        self.joint_state_received.connect(self.update_joint_display)

        self.init_ros()

    def init_ui(self):
        """初始化 GUI 界面"""
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ===== 标题标签 =====
        title_label = QLabel("AU2Joint 编辑器 - 选择单元格并保存关节角度")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # ===== 说明标签 =====
        info_label = QLabel(
            "使用说明: 点击表格中的单元格选择 AU 和强度，然后点击'保存'按钮记录当前关节角度"
        )
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        # ===== 表格 =====
        # 行数 = 1 (AU_DESCRIPTION 说明行) + INTENSITY_COUNT (强度数据行)
        self.table = QTableWidget()
        self.table.setRowCount(1 + INTENSITY_COUNT)
        self.table.setColumnCount(AU_COUNT)

        # 设置水平表头为 AU_NAMES
        self.table.setHorizontalHeaderLabels(AU_NAMES)

        # 设置垂直表头：第一行为 AU说明，后面为强度 1-5
        vertical_headers = ["AU说明"] + [f"强度 {i}" for i in INTENSITY_LEVELS]
        self.table.setVerticalHeaderLabels(vertical_headers)

        # 设置表格样式
        for col in range(AU_COUNT):
            self.table.setColumnWidth(col, 120)
        self.table.setRowHeight(0, 35)
        for row in range(1, 1 + INTENSITY_COUNT):
            self.table.setRowHeight(row, 50)

        # 表头样式
        header_font = QFont()
        header_font.setBold(True)
        self.table.horizontalHeader().setFont(header_font)
        self.table.verticalHeader().setFont(header_font)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)

        # 禁止编辑，只允许选择
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectItems)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        # 填充 AU_DESCRIPTION 行 (row 0)
        for col in range(AU_COUNT):
            item = QTableWidgetItem(AU_DESCRIPTION[col])
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(Qt.ItemIsEnabled)  # 不可选
            item.setBackground(QBrush(QColor(230, 230, 250)))  # 淡紫色背景
            desc_font = QFont()
            desc_font.setPointSize(9)
            desc_font.setItalic(True)
            item.setFont(desc_font)
            self.table.setItem(0, col, item)

        # 连接单元格点击信号
        self.table.cellClicked.connect(self.on_cell_selected)

        main_layout.addWidget(self.table)

        # ===== 状态标签 =====
        self.status_label = QLabel("请选择一个单元格")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(11)
        self.status_label.setFont(status_font)
        main_layout.addWidget(self.status_label)

        # ===== 关节角度显示标签 =====
        self.joints_label = QLabel("等待 ROS2 消息...")
        self.joints_label.setAlignment(Qt.AlignCenter)
        self.joints_label.setWordWrap(True)
        main_layout.addWidget(self.joints_label)

        # ===== 按钮区域 =====
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # 保存按钮
        self.save_button = QPushButton("保存")
        self.save_button.setMinimumHeight(50)
        self.save_button.setFont(header_font)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.on_save_clicked)
        button_layout.addWidget(self.save_button)

        # 播放按钮
        self.play_button = QPushButton("播放")
        self.play_button.setMinimumHeight(50)
        self.play_button.setFont(header_font)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.on_play_clicked)
        button_layout.addWidget(self.play_button)

        main_layout.addLayout(button_layout)

        # ===== 文件路径标签 =====
        path_label = QLabel(f"数据文件: {DATA_FILE_PATH}")
        path_label.setAlignment(Qt.AlignCenter)
        path_label.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(path_label)

    def init_data(self):
        """初始化数据数组，加载已有文件或创建新数组"""
        if os.path.exists(DATA_FILE_PATH):
            try:
                self.data = np.load(DATA_FILE_PATH)
                self._log_info(f"已加载数据文件: {DATA_FILE_PATH}, 形状: {self.data.shape}")

                # 验证形状
                if self.data.shape != (AU_COUNT, INTENSITY_COUNT, JOINT_ANGLE_COUNT):
                    self._log_warn(
                        f"数据形状不匹配，重新初始化。 "
                        f"期望: {(AU_COUNT, INTENSITY_COUNT, JOINT_ANGLE_COUNT)}, "
                        f"实际: {self.data.shape}"
                    )
                    self.data = np.full(
                        (AU_COUNT, INTENSITY_COUNT, JOINT_ANGLE_COUNT),
                        DEFAULT_ANGLE_VALUE,
                        dtype=np.float64,
                    )
                else:
                    # 加载成功，更新表格显示
                    self.update_table_display()

            except Exception as e:
                self._log_err(f"加载数据文件失败: {e}")
                self.data = np.full(
                    (AU_COUNT, INTENSITY_COUNT, JOINT_ANGLE_COUNT),
                    DEFAULT_ANGLE_VALUE,
                    dtype=np.float64,
                )
        else:
            self._log_info(f"数据文件不存在，创建新数组: {DATA_FILE_PATH}")
            self.data = np.full(
                (AU_COUNT, INTENSITY_COUNT, JOINT_ANGLE_COUNT),
                DEFAULT_ANGLE_VALUE,
                dtype=np.float64,
            )
            # 确保目录存在
            os.makedirs(CONFIG_DIR, exist_ok=True)
            self.save_data()

    def init_ros(self):
        """初始化 ROS2 节点和订阅者"""
        try:
            rclpy.init()
            self.ros_node = rclpy.create_node("au2joint_editor")
            self._log_info("ROS2 节点已启动: au2joint_editor")

            # 订阅关节状态话题
            self.joint_state_sub = self.ros_node.create_subscription(
                JointState,
                "/joint_state_blender",
                self.joint_state_callback,
                10,
            )
            self._log_info("已订阅话题: /joint_state_blender")

            # 启动 ROS2 监听线程
            self.ros_thread = ROS2ListenerThread(self.ros_node)
            self.ros_thread.start()

        except Exception as e:
            self._log_err(f"ROS2 初始化失败: {e}")
            QMessageBox.critical(self, "ROS2 错误", f"ROS2 初始化失败:\n{e}")

    def _log_info(self, msg):
        """输出 info 级别日志"""
        if self.ros_node:
            self.ros_node.get_logger().info(msg)
        else:
            print(f"[INFO] {msg}")

    def _log_warn(self, msg):
        """输出 warn 级别日志"""
        if self.ros_node:
            self.ros_node.get_logger().warn(msg)
        else:
            print(f"[WARN] {msg}")

    def _log_err(self, msg):
        """输出 error 级别日志"""
        if self.ros_node:
            self.ros_node.get_logger().error(msg)
        else:
            print(f"[ERROR] {msg}")

    def update_table_display(self):
        """根据数据更新表格显示，标记已保存的单元格"""
        for au_idx in range(AU_COUNT):
            for intensity_idx in range(INTENSITY_COUNT):
                # 表格行号 = intensity_idx + 1 (跳过 AU_DESCRIPTION 行)
                table_row = intensity_idx + 1

                # 检查该单元格是否已保存数据（不是默认值）
                angles = self.data[au_idx, intensity_idx]
                has_data = not np.allclose(angles, DEFAULT_ANGLE_VALUE)

                item = QTableWidgetItem()
                if has_data:
                    item.setText("已保存")
                    item.setBackground(QBrush(QColor(200, 255, 200)))  # 浅绿色
                else:
                    item.setText("-")
                    item.setBackground(QBrush(QColor(255, 255, 255)))  # 白色

                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(table_row, au_idx, item)

    def on_cell_selected(self, row, column):
        """单元格被选中时的回调"""
        # 跳过 AU_DESCRIPTION 行 (row 0)
        if row == 0:
            self.status_label.setText("AU说明行不可选择，请点击下方的强度单元格")
            self.save_button.setEnabled(False)
            self.play_button.setEnabled(False)
            return

        # 强度索引 = 表格行号 - 1
        self.selected_intensity_idx = row - 1
        self.selected_au_idx = column

        au_name = AU_NAMES[column]
        intensity = INTENSITY_LEVELS[self.selected_intensity_idx]

        self.status_label.setText(f"已选择: AU={au_name}, 强度={intensity}")

        # 启用按钮
        self.save_button.setEnabled(True)
        self.play_button.setEnabled(True)

        # 如果该位置已有数据，显示数据
        angles = self.data[column, self.selected_intensity_idx]
        has_data = not np.allclose(angles, DEFAULT_ANGLE_VALUE)

        if has_data:
            angles_str = ", ".join([f"{a:.4f}" for a in angles])
            self.joints_label.setText(f"当前存储的关节角度: [{angles_str}]")
        else:
            self.joints_label.setText("该位置暂无数据")

    def joint_state_callback(self, msg):
        """ROS2 JointState 回调函数"""
        self.latest_joint_state = msg

        # 提取位置数据
        positions = list(msg.position)

        # 如果关节角度数量不等于14，给出警告
        if len(positions) != JOINT_ANGLE_COUNT:
            self._log_warn(
                f"关节数量不匹配: 期望 {JOINT_ANGLE_COUNT}, 实际 {len(positions)}"
            )

        # 截断或填充到14个
        if len(positions) >= JOINT_ANGLE_COUNT:
            positions = positions[:JOINT_ANGLE_COUNT]
        else:
            positions = positions + [DEFAULT_ANGLE_VALUE] * (JOINT_ANGLE_COUNT - len(positions))

        # 发射信号更新 GUI（线程安全）
        self.joint_state_received.emit(positions)

    def update_joint_display(self, positions):
        """更新关节角度显示（槽函数）"""
        angles_str = ", ".join([f"{p:.4f}" for p in positions])
        if self.selected_au_idx >= 0 and self.selected_intensity_idx >= 0:
            self.joints_label.setText(
                f"当前关节角度 (来自 ROS2): [{angles_str}]"
            )
        else:
            self.joints_label.setText(
                f"收到关节角度 (请先选择单元格): [{angles_str}]"
            )

    def on_save_clicked(self):
        """保存按钮点击事件"""
        if self.selected_au_idx < 0 or self.selected_intensity_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择一个单元格")
            return

        if self.latest_joint_state is None:
            QMessageBox.warning(
                self, "警告", "尚未收到 ROS2 话题 /joint_state_blender 的消息"
            )
            return

        # 提取关节角度
        positions = list(self.latest_joint_state.position)

        if len(positions) >= JOINT_ANGLE_COUNT:
            positions = positions[:JOINT_ANGLE_COUNT]
        else:
            positions = positions + [DEFAULT_ANGLE_VALUE] * (JOINT_ANGLE_COUNT - len(positions))

        # 保存到数据数组
        self.data[self.selected_au_idx, self.selected_intensity_idx] = positions

        # 保存到文件
        self.save_data()

        # 更新表格显示
        self.update_table_display()

        # 更新状态
        au_name = AU_NAMES[self.selected_au_idx]
        intensity = INTENSITY_LEVELS[self.selected_intensity_idx]
        self.status_label.setText(f"已保存到: AU={au_name}, 强度={intensity}")

        # 显示保存的关节角度
        angles_str = ", ".join([f"{p:.4f}" for p in positions])
        self.joints_label.setText(f"已保存关节角度: [{angles_str}]")

        self._log_info(
            f"已保存: AU={au_name}, 强度={intensity}, 角度={positions}"
        )

    def on_play_clicked(self):
        """播放按钮点击事件 - 暂时不做处理"""
        # TODO: 实现播放功能
        pass

    def save_data(self):
        """保存数据到 .npy 文件"""
        try:
            # 确保目录存在
            os.makedirs(CONFIG_DIR, exist_ok=True)

            # 保存数据
            np.save(DATA_FILE_PATH, self.data)
            self._log_info(f"数据已保存到: {DATA_FILE_PATH}")

        except Exception as e:
            self._log_err(f"保存数据失败: {e}")
            QMessageBox.critical(self, "保存错误", f"保存数据失败:\n{e}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 保存数据
        self.save_data()

        # 关闭 ROS2
        if self.ros_thread and self.ros_thread.isRunning():
            rclpy.shutdown()
            self.ros_thread.wait(1000)

        event.accept()


def main():
    """主函数"""
    # 创建 Qt 应用
    app = QApplication(sys.argv)

    # 创建主窗口
    editor = AU2JointEditor()
    editor.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
