import os
import cv2
from skimage.metrics import structural_similarity as ssim
from scenedetect import detect, ContentDetector, SceneDetector, split_video_ffmpeg

class SSIMDetector(SceneDetector):
    """基于结构相似性（SSIM）的场景检测器。
    
    比较相邻帧的灰度结构相似性，低于阈值时判定为场景跳变。
    """

    def __init__(self, threshold=0.65, min_scene_len=15):
        super().__init__()
        self.threshold = threshold          # SSIM 低于此值触发跳变
        self.min_scene_len = min_scene_len  # 最小场景长度（帧）
        self._last_frame = None
        self._last_cut = None

    def process_frame(self, frame_num, frame_img):
        # 新版 PySceneDetect 传入 FrameTimecode，旧版传入整数帧号。
        position = frame_num.get_frames() if hasattr(frame_num, "get_frames") else frame_num
        gray_curr = (
            cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            if frame_img.ndim == 3 else frame_img
        )
        if self._last_frame is None:
            self._last_frame = gray_curr.copy()
            self._last_cut = position
            return []

        # 计算 SSIM（1.0 表示完全相同，越低差异越大）
        score = ssim(self._last_frame, gray_curr, data_range=255)
        self._last_frame = gray_curr.copy()

        if score < self.threshold and position - self._last_cut >= self.min_scene_len:
            self._last_cut = position
            # 保留调用方传入的位置类型，不需要额外传入帧率。
            return [frame_num]
        return []

    def is_processing_required(self):
        return True


def split_video_ssim(video_path, output_dir, threshold=0.6):
    """使用 SSIM 检测器切分视频，接口与 split_video 完全一致。

    Args:
        video_path: 输入视频路径。
        output_dir: 输出目录。
        threshold: 0~1 时直接作为 SSIM 阈值（如配置中的 0.6），
                   数值越大越敏感；大于 1 且不超过 100 时作为差异百分比，
                   映射为 1 - threshold/100（默认 27.0 对应 SSIM 0.73）。
    """

    scenes = detect(
        video_path,
        SSIMDetector(
            threshold=threshold,
            min_scene_len=15
        ),
        start_in_scene=True,
    )

    if not scenes:
        print("未检测到有效场景")
        return

    os.makedirs(output_dir, exist_ok=True)
    split_video_ffmpeg(
        video_path,
        scenes,
        output_dir=output_dir
    )

    return scenes

def split_video_hsv(video_path, output_dir, threshold=27.0):

    # 检测视频场景
    scenes = detect(
        video_path,
        ContentDetector(
            threshold=threshold,
            min_scene_len=15
        )
    )

    if not scenes:
        print("未检测到有效场景")
        return

    # 按照场景切分视频
    split_video_ffmpeg(
        video_path,
        scenes,
        output_dir=output_dir
    )

    return scenes
