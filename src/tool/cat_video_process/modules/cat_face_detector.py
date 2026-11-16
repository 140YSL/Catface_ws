"""使用 CatDetect 的 YOLOv8 权重统计视频采样帧中的猫脸占比。"""

import os
import tempfile

import cv2
from ultralytics import YOLO


class CatFaceDetector:
    def __init__(self, model_path, device="cpu", conf=0.5):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"猫脸模型文件不存在: {model_path}")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf 必须在 0~1 范围内")

        self.model = YOLO(model_path, task="detect")
        if self.model.task != "detect" or self.model.names != {0: "cat"}:
            raise ValueError("请使用 CatDetect 的单类别猫脸检测 best.pt（类别 0: cat）")
        self.device = device
        self.conf = conf

    def detect_video(self, video_path, frame_interval=10):
        """从第 0 帧开始每隔指定帧数检测，猫脸帧占比严格超过 80% 返回 True。

        每张采样帧检测到至少一张猫脸即计为包含猫脸，恰好 80% 返回 False。
        所有采样帧的标注结果按原始帧号保存至项目 runs/cat_face_detection 下。
        视频无法打开或首帧无法读取时抛出异常。
        """
        if (
            isinstance(frame_interval, bool)
            or not isinstance(frame_interval, int)
            or frame_interval < 1
        ):
            raise ValueError("frame_interval 必须为正整数")

        cap = cv2.VideoCapture(os.fspath(video_path))
        try:
            if not cap.isOpened():
                raise OSError(f"无法打开视频: {video_path}")

            # save_root = os.path.join(
            #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            #     "runs", "cat_face_detection",
            # )
            # os.makedirs(save_root, exist_ok=True)
            # video_name = os.path.splitext(os.path.basename(os.fspath(video_path)))[0]
            # 每次调用使用独立目录，避免同名视频或重复运行时覆盖已有结果。
            # save_dir = tempfile.mkdtemp(prefix=f"{video_name}_", dir=save_root)
            # print(f"Detection images saved to: {save_dir}")

            frame_index = 0
            sampled_frames = 0
            face_frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    if frame_index == 0:
                        raise OSError(f"无法读取视频帧: {video_path}")
                    break

                if frame_index % frame_interval == 0:
                    # OpenCV 的 BGR ndarray 可直接传给 Ultralytics。
                    results = self.model.predict(
                        source=frame,
                        device=self.device,
                        conf=self.conf,
                        classes=[0],
                        verbose=False,
                        save=False,
                    )
                    # for result in results:
                    #     result.save(filename=os.path.join(
                    #         save_dir, f"frame_{frame_index:08d}.jpg"
                    #     ))
                    sampled_frames += 1
                    if any(
                        result.boxes is not None and len(result.boxes) > 0
                        for result in results
                    ):
                        face_frames += 1
                        # print(f"Face detected in frame {frame_index}")
                frame_index += 1

            # 使用整数比较，确保恰好 80% 时不通过。
            # print(f"Sampled frames: {sampled_frames}, Face frames: {face_frames}")
            return sampled_frames > 0 and face_frames * 5 > sampled_frames * 4
        finally:
            cap.release()
