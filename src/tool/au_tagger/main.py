#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QFileInfo, QTimer, Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFileSystemModel,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSlider,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

try:
    from .annotation_store import AU_DESCRIPTION, AnnotationStore
except ImportError:
    from annotation_store import AU_DESCRIPTION, AnnotationStore


VIDEO_EXTENSIONS = [
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.mkv",
    "*.wmv",
    "*.flv",
    "*.webm",
    "*.m4v",
    "*.mpeg",
    "*.mpg",
]


def format_time(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class AUTaggerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.capture = None
        self.video_path = None
        self.annotation_store = None
        self.intensity_array = None
        self.current_frame = -1
        self.frame_count = 0
        self.fps = 25.0
        self.current_image = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.show_next_frame)

        self.build_ui()
        self.build_shortcuts()
        self.set_video_controls_enabled(False)

    def build_ui(self):
        self.setWindowTitle("视频 AU 人工标注工具")
        self.resize(1400, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_file_panel())
        splitter.addWidget(self.create_player_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1100])
        main_layout.addWidget(splitter, 1)
        main_layout.addWidget(self.create_annotation_panel())

    def create_file_panel(self):
        panel = QGroupBox("文件树")
        layout = QVBoxLayout(panel)

        choose_folder_button = QPushButton("打开文件夹")
        choose_folder_button.clicked.connect(self.choose_folder)
        layout.addWidget(choose_folder_button)

        self.folder_label = QLabel(str(Path.cwd()))
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        self.file_model = QFileSystemModel(self)
        self.file_model.setNameFilters(VIDEO_EXTENSIONS)
        self.file_model.setNameFilterDisables(False)
        self.file_model.setRootPath(str(Path.cwd()))

        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(str(Path.cwd())))
        self.file_tree.setHeaderHidden(True)
        self.file_tree.doubleClicked.connect(self.on_file_activated)
        for column in range(1, self.file_model.columnCount()):
            self.file_tree.hideColumn(column)
        layout.addWidget(self.file_tree, 1)

        hint = QLabel("双击视频文件打开")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)
        return panel

    def create_player_panel(self):
        panel = QGroupBox("视频播放")
        layout = QVBoxLayout(panel)

        self.video_name_label = QLabel("尚未打开视频")
        self.video_name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_name_label)

        self.video_label = QLabel("请从左侧文件树中选择视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #151515; color: #bbbbbb;")
        layout.addWidget(self.video_label, 1)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.sliderMoved.connect(self.seek_to_frame)
        layout.addWidget(self.position_slider)

        controls = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)

        self.frame_label = QLabel("帧: 0 / 0")
        self.frame_label.setMinimumWidth(150)
        controls.addWidget(self.frame_label)
        controls.addStretch(1)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        shortcut_hint = QLabel("快捷键: Space 播放/暂停    ← 后退一帧    → 前进一帧")
        shortcut_hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(shortcut_hint)
        return panel

    def create_annotation_panel(self):
        panel = QGroupBox("AU 强度")
        panel_layout = QHBoxLayout(panel)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setMinimumHeight(225)

        slider_widget = QWidget()
        slider_layout = QGridLayout(slider_widget)
        slider_layout.setColumnStretch(1, 1)
        slider_layout.setColumnStretch(6, 1)
        self.au_sliders = []
        self.au_value_labels = []
        self.au_insert_buttons = []
        self.au_modify_buttons = []

        for index, description in enumerate(AU_DESCRIPTION):
            row = index // 2
            first_column = (index % 2) * 5
            description_label = QLabel(description)
            description_label.setMinimumWidth(130)
            # 滑块内部单位为 0.2 强度，0..25 对应强度 0.0..5.0
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 25)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider.setTickInterval(5)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setValue(0)
            value_label = QLabel("0.0")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setMinimumWidth(32)
            slider.valueChanged.connect(
                lambda value, label=value_label: label.setText(f"{value / 5:.1f}")
            )
            insert_button = QPushButton("插帧")
            insert_button.clicked.connect(
                lambda checked=False, au_index=index: self.insert_au(au_index)
            )
            modify_button = QPushButton("修改")
            modify_button.clicked.connect(
                lambda checked=False, au_index=index: self.modify_au(au_index)
            )
            slider_layout.addWidget(description_label, row, first_column)
            slider_layout.addWidget(slider, row, first_column + 1)
            slider_layout.addWidget(value_label, row, first_column + 2)
            slider_layout.addWidget(insert_button, row, first_column + 3)
            slider_layout.addWidget(modify_button, row, first_column + 4)
            self.au_sliders.append(slider)
            self.au_value_labels.append(value_label)
            self.au_insert_buttons.append(insert_button)
            self.au_modify_buttons.append(modify_button)

        scroll_area.setWidget(slider_widget)
        panel_layout.addWidget(scroll_area, 1)

        action_layout = QVBoxLayout()
        self.save_button = QPushButton("保存")
        self.save_button.setMinimumSize(100, 48)
        self.save_button.clicked.connect(self.save_annotations)
        action_layout.addWidget(self.save_button)

        self.annotation_status_label = QLabel("未加载标注文件")
        self.annotation_status_label.setWordWrap(True)
        self.annotation_status_label.setMaximumWidth(230)
        action_layout.addWidget(self.annotation_status_label)
        action_layout.addStretch(1)
        panel_layout.addLayout(action_layout)
        return panel

    def build_shortcuts(self):
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.setContext(Qt.ApplicationShortcut)
        self.left_shortcut.activated.connect(self.show_previous_frame)

        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.setContext(Qt.ApplicationShortcut)
        self.right_shortcut.activated.connect(self.show_next_frame)

        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.setContext(Qt.ApplicationShortcut)
        self.space_shortcut.activated.connect(self.toggle_playback)

    def choose_folder(self):
        default_dir = (Path(__file__).resolve().parent / "../../../cat_video").resolve()
        current_dir = self.folder_label.text()
        start_dir = str(default_dir) if default_dir.exists() and current_dir == str(Path.cwd()) else current_dir
        folder = QFileDialog.getExistingDirectory(
            self, "选择视频文件夹", start_dir
        )
        if not folder:
            return
        self.file_model.setRootPath(folder)
        self.file_tree.setRootIndex(self.file_model.index(folder))
        self.folder_label.setText(folder)

    def on_file_activated(self, model_index):
        path = QFileInfo(self.file_model.filePath(model_index))
        if path.isFile():
            self.load_video(path.absoluteFilePath())

    def load_video(self, path):
        self.pause_playback()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            QMessageBox.critical(self, "打开失败", f"无法打开视频:\n{path}")
            return

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            capture.release()
            QMessageBox.critical(self, "打开失败", "无法读取视频帧数")
            return
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        if self.capture is not None:
            self.capture.release()
        self.capture = capture
        self.video_path = Path(path)
        self.frame_count = frame_count
        self.fps = fps
        self.current_frame = -1
        self.play_timer.setInterval(max(1, round(1000.0 / self.fps)))

        self.video_name_label.setText(self.video_path.name)
        self.position_slider.setRange(0, self.frame_count - 1)

        self.intensity_array = np.zeros(
            (self.frame_count, len(AU_DESCRIPTION)), dtype=np.float64
        )
        annotation_path = self.video_path.with_suffix(".npz")
        try:
            self.annotation_store = AnnotationStore.load(annotation_path)
        except (OSError, ValueError) as error:
            self.annotation_store = None
            self.set_video_controls_enabled(True)
            self.set_annotation_controls_enabled(False)
            self.annotation_status_label.setText("标注文件读取失败，标注已禁用")
            QMessageBox.critical(
                self,
                "标注文件错误",
                f"无法读取 {annotation_path}:\n{error}\n\n"
                "为避免覆盖原文件，当前视频已禁用标注。",
            )
        else:
            for frame_index in range(self.frame_count):
                record = self.annotation_store.get_latest_before(frame_index)
                if record is not None:
                    self.intensity_array[frame_index] = record[1]
            self.set_video_controls_enabled(True)
            self.update_annotation_status()

        if not self.display_frame(0):
            QMessageBox.critical(self, "读取失败", "无法读取视频的第一帧")

    def set_annotation_controls_enabled(self, enabled):
        self.save_button.setEnabled(enabled)
        for slider in self.au_sliders:
            slider.setEnabled(enabled)
        for button in self.au_insert_buttons + self.au_modify_buttons:
            button.setEnabled(enabled)

    def set_video_controls_enabled(self, enabled):
        self.play_button.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)
        self.set_annotation_controls_enabled(enabled)

    def display_frame(self, frame_index):
        if self.capture is None or self.frame_count == 0:
            return False

        frame_index = max(0, min(int(frame_index), self.frame_count - 1))
        if self.current_frame < 0 or frame_index != self.current_frame + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self.capture.read()
        if not success:
            self.pause_playback()
            return False

        self.current_frame = frame_index
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        self.current_image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self.update_video_pixmap()

        self.position_slider.blockSignals(True)
        self.position_slider.setValue(frame_index)
        self.position_slider.blockSignals(False)
        self.frame_label.setText(f"帧: {frame_index + 1} / {self.frame_count}")
        self.time_label.setText(
            f"{format_time(frame_index / self.fps)} / "
            f"{format_time((self.frame_count - 1) / self.fps)}"
        )
        self.restore_annotation()
        return True

    def update_video_pixmap(self):
        if self.current_image is None:
            return
        pixmap = QPixmap.fromImage(self.current_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def restore_annotation(self):
        values = np.zeros(len(AU_DESCRIPTION), dtype=np.float64)
        if self.intensity_array is not None and 0 <= self.current_frame < len(
            self.intensity_array
        ):
            values = self.intensity_array[self.current_frame]
        for slider, value in zip(self.au_sliders, values):
            slider.setValue(int(round(float(value) * 5)))

    def seek_to_frame(self, frame_index):
        self.pause_playback()
        self.display_frame(frame_index)

    def show_previous_frame(self):
        if self.capture is None:
            return
        self.pause_playback()
        self.display_frame(self.current_frame - 1)

    def show_next_frame(self):
        if self.capture is None:
            return
        if self.current_frame >= self.frame_count - 1:
            self.pause_playback()
            return
        self.display_frame(self.current_frame + 1)

    def toggle_playback(self):
        if self.capture is None:
            return
        if self.play_timer.isActive():
            self.pause_playback()
            return
        if self.current_frame >= self.frame_count - 1:
            if not self.display_frame(0):
                return
        self.play_button.setText("暂停")
        self.play_timer.start()

    def pause_playback(self):
        self.play_timer.stop()
        self.play_button.setText("播放")

    def insert_au(self, au_index):
        if (
            self.capture is None
            or self.intensity_array is None
            or self.current_frame < 0
        ):
            return

        value = self.au_sliders[au_index].value() / 5
        self.intensity_array[self.current_frame:, au_index] = value
        self.annotation_status_label.setText(
            f"已插帧: {AU_DESCRIPTION[au_index]}\n"
            f"第 {self.current_frame + 1} 帧起 = {value:.1f}"
        )

    def modify_au(self, au_index):
        if (
            self.capture is None
            or self.intensity_array is None
            or self.current_frame < 0
        ):
            return

        value = self.au_sliders[au_index].value() / 5
        self.intensity_array[self.current_frame, au_index] = value
        self.annotation_status_label.setText(
            f"已修改: {AU_DESCRIPTION[au_index]}\n"
            f"第 {self.current_frame + 1} 帧 = {value:.1f}"
        )

    def save_annotations(self):
        if (
            self.capture is None
            or self.intensity_array is None
            or self.video_path is None
        ):
            return

        records = {
            frame_index: (
                frame_index / self.fps,
                self.intensity_array[frame_index].copy(),
            )
            for frame_index in range(self.frame_count)
        }
        annotation_path = self.video_path.with_suffix(".npz")
        try:
            AnnotationStore(annotation_path, records).save()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "保存失败", f"无法保存标注:\n{error}")
            return

        try:
            self.annotation_store = AnnotationStore.load(annotation_path)
        except (OSError, ValueError):
            self.annotation_store = None
        self.update_annotation_status()
        self.annotation_status_label.setText(
            f"已保存到 {annotation_path.name}\n"
            f"共 {self.frame_count} 帧"
        )

    def update_annotation_status(self):
        if self.annotation_store is None:
            return
        self.annotation_status_label.setText(
            f"{self.annotation_store.path.name}\n"
            f"已有 {len(self.annotation_store)} 个标注帧"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_video_pixmap()

    def closeEvent(self, event):
        self.pause_playback()
        if self.capture is not None:
            self.capture.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = AUTaggerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
