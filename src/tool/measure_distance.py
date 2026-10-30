#!/usr/bin/env python3

import sys
import math
import os

# ============================================================
# 必须在导入和初始化 Qt 应用之前设置高 DPI 环境
# 目标：让 Qt 使用与 X11 鼠标事件一致的屏幕像素坐标
# ============================================================

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

from PyQt5.QtCore import (
    Qt,
    QRect,
    QPoint,
    pyqtSignal,
    QObject,
)

from PyQt5.QtGui import (
    QPainter,
    QPen,
    QColor,
    QFont,
    QGuiApplication,
)

from PyQt5.QtWidgets import QApplication, QWidget

from pynput import mouse, keyboard


# ============================================================
# 测量颜色：每组两个点使用相同颜色
# ============================================================

COLORS = [
    QColor(255, 60, 60),       # 红色
    QColor(50, 220, 100),      # 绿色
    QColor(50, 160, 255),      # 蓝色
    QColor(255, 210, 40),      # 黄色
    QColor(220, 80, 255),      # 紫色
    QColor(40, 230, 220),      # 青色
    QColor(255, 150, 50),      # 橙色
]


class EventBridge(QObject):
    """将后台鼠标和键盘事件安全地发送到 Qt 主线程。"""

    mouse_clicked = pyqtSignal(int, int)
    escape_pressed = pyqtSignal()


class MeasurementOverlay(QWidget):

    def __init__(self):
        super().__init__()

        # 所有已完成的测量组
        # 每组数据永久保留，直到程序退出
        self.measurements = []

        # 当前组已经选择的第一个点
        self.pending_point = None

        # ----------------------------------------------------
        # 获取所有屏幕的虚拟桌面区域
        # ----------------------------------------------------

        screens = QGuiApplication.screens()

        if not screens:
            raise RuntimeError("没有检测到可用的显示器。")

        self.desktop_rect = QRect()

        for screen in screens:
            self.desktop_rect = self.desktop_rect.united(
                screen.geometry()
            )

        if self.desktop_rect.isEmpty():
            raise RuntimeError("无法获取有效的桌面区域。")

        # ----------------------------------------------------
        # 创建信号桥接
        # ----------------------------------------------------

        self.bridge = EventBridge()

        self.bridge.mouse_clicked.connect(
            self.handle_click
        )

        self.bridge.escape_pressed.connect(
            self.close
        )

        # ----------------------------------------------------
        # 创建覆盖整个桌面的透明绘图层
        # ----------------------------------------------------

        self.setGeometry(self.desktop_rect)

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground,
            True
        )

        self.setAttribute(
            Qt.WA_TransparentForMouseEvents,
            True
        )

        self.setAttribute(
            Qt.WA_ShowWithoutActivating,
            True
        )

        self.show()

        print("=" * 45)
        print("桌面像素测距工具已启动")
        print(f"桌面区域：{self.desktop_rect}")
        print("左键依次点击两个点，完成一组测量。")
        print("每组使用不同颜色。")
        print("历史标记、连线和距离文字永久保留。")
        print("按 ESC 退出。")
        print("=" * 45)
        print()

    # ========================================================
    # 坐标转换：X11 全局坐标 -> Qt 绘图层局部坐标
    # ========================================================

    def global_to_local(self, x, y):
        """
        将 pynput 获取的全局屏幕坐标转换成绘图层局部坐标。

        使用 mapFromGlobal，而不是手动减去桌面原点。
        """

        local_point = self.mapFromGlobal(
            QPoint(int(x), int(y))
        )

        return local_point.x(), local_point.y()

    # ========================================================
    # 处理鼠标点击
    # ========================================================

    def handle_click(self, x, y):

        # ----------------------------------------------------
        # 第一次点击：开始一组测量
        # ----------------------------------------------------

        if self.pending_point is None:

            self.pending_point = (x, y)

            group_index = len(self.measurements) + 1

            print(
                f"第 {group_index} 组："
                f"第一个点 ({x}, {y})"
            )

            print("等待第二个点……")

        # ----------------------------------------------------
        # 第二次点击：完成一组测量
        # ----------------------------------------------------

        else:

            x1, y1 = self.pending_point
            x2, y2 = x, y

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            distance = math.hypot(
                x2 - x1,
                y2 - y1
            )

            color = COLORS[
                len(self.measurements) % len(COLORS)
            ]

            self.measurements.append({
                "p1": (x1, y1),
                "p2": (x2, y2),
                "dx": dx,
                "dy": dy,
                "distance": distance,
                "color": color,
            })

            print(f"第二个点：({x2}, {y2})")
            print(f"水平距离：{dx} px")
            print(f"垂直距离：{dy} px")
            print(f"直线距离：{distance:.2f} px")
            print("-" * 45)

            # 当前组结束，下一次点击开始新的一组
            self.pending_point = None

            print("请继续点击下一组点，或按 ESC 退出。")
            print()

        # 刷新绘图层
        self.update()

    # ========================================================
    # 绘制所有高亮点、连线和距离标注
    # ========================================================

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.Antialiasing,
            True
        )

        # ----------------------------------------------------
        # 绘制所有已经完成的测量组
        # ----------------------------------------------------

        for item in self.measurements:

            # 将两个全局屏幕坐标转换为绘图层局部坐标
            p1 = self.global_to_local(
                item["p1"][0],
                item["p1"][1]
            )

            p2 = self.global_to_local(
                item["p2"][0],
                item["p2"][1]
            )

            color = item["color"]

            # 1. 绘制连接线
            pen = QPen(color, 1.5)

            pen.setCapStyle(Qt.RoundCap)

            painter.setPen(pen)

            painter.drawLine(
                p1[0],
                p1[1],
                p2[0],
                p2[1]
            )

            # 2. 绘制两个端点
            self.draw_point(
                painter,
                p1,
                color
            )

            self.draw_point(
                painter,
                p2,
                color
            )

            # 3. 计算线段中点
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            length = math.hypot(dx, dy)

            # 计算线段法向量，用于将文字放在线段旁边
            if length > 0:

                nx = -dy / length
                ny = dx / length

            else:

                nx = 0
                ny = -1

            # 文字偏移距离
            offset = 18

            tx = mx + nx * offset
            ty = my + ny * offset

            label = f"{item['distance']:.2f} px"

            # 4. 绘制距离文字
            painter.setFont(
                QFont(
                    "Sans Serif",
                    10,
                    QFont.Bold
                )
            )

            metrics = painter.fontMetrics()

            text_width = metrics.horizontalAdvance(label)
            text_height = metrics.height()

            rect = QRect(
                int(tx - text_width / 2 - 5),
                int(ty - text_height / 2 - 3),
                text_width + 10,
                text_height + 6
            )

            # # 半透明黑色背景
            # painter.setPen(Qt.NoPen)

            # painter.setBrush(
            #     QColor(0, 0, 0, 180)
            # )

            # painter.drawRoundedRect(
            #     rect,
            #     4,
            #     4
            # )

            # # 文字颜色与对应测量组的连线相同
            # painter.setPen(color)

            # painter.drawText(
            #     rect,
            #     Qt.AlignCenter,
            #     label
            # )

        # ----------------------------------------------------
        # 如果当前只选择了第一个点，显示临时高亮
        # ----------------------------------------------------

        if self.pending_point is not None:

            point = self.global_to_local(
                self.pending_point[0],
                self.pending_point[1]
            )

            color = COLORS[
                len(self.measurements) % len(COLORS)
            ]

            self.draw_point(
                painter,
                point,
                color
            )

        painter.end()

    # ========================================================
    # 绘制高亮点
    # ========================================================

    @staticmethod
    def draw_point(painter, point, color):

        x, y = point

        # 白色外圈
        painter.setPen(
            QPen(
                QColor(255, 255, 255),
                2
            )
        )

        painter.setBrush(color)

        painter.drawEllipse(
            int(x - 5),
            int(y - 5),
            10,
            10
        )

        # 中心圆点
        painter.setPen(Qt.NoPen)

        painter.setBrush(color)

        painter.drawEllipse(
            int(x - 2),
            int(y - 2),
            4,
            4
        )


# ============================================================
# 主程序
# ============================================================

def main():

    # 必须在 QApplication 创建之前关闭 Qt 高 DPI 缩放
    QApplication.setAttribute(
        Qt.AA_DisableHighDpiScaling,
        True
    )

    app = QApplication(sys.argv)

    # 关闭最后一个窗口时退出应用
    app.setQuitOnLastWindowClosed(True)

    # 创建透明绘图层
    overlay = MeasurementOverlay()

    # --------------------------------------------------------
    # 全局鼠标监听
    # --------------------------------------------------------

    def on_click(x, y, button, pressed):

        if pressed and button == mouse.Button.left:

            # pynput 在 X11 下提供的全局鼠标坐标
            overlay.bridge.mouse_clicked.emit(
                int(x),
                int(y)
            )

    # --------------------------------------------------------
    # 全局键盘监听
    # --------------------------------------------------------

    def on_press(key):

        if key == keyboard.Key.esc:

            overlay.bridge.escape_pressed.emit()

            # 停止键盘监听
            return False

    # 创建监听器
    mouse_listener = mouse.Listener(
        on_click=on_click
    )

    keyboard_listener = keyboard.Listener(
        on_press=on_press
    )

    # 启动监听器
    mouse_listener.start()
    keyboard_listener.start()

    try:

        exit_code = app.exec_()

    except KeyboardInterrupt:

        exit_code = 0

    finally:

        mouse_listener.stop()
        keyboard_listener.stop()

        print("\n测量程序已退出。")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()