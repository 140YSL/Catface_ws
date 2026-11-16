import os

# 原始视频目录
INPUT_DIR = "./../../../cat_video/raw"

# 临时视频目录
TEMP_DIR = "./../../../cat_video/temp"

# 最终视频目录
OUTPUT_DIR = "./../../../cat_video/final"

# 猫脸检测模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight", "cat_face.pt")

# 最短视频时长（秒）
MIN_DURATION = 5.0

# 视频场景检测阈值
SCENE_THRESHOLD = 0.6

# 猫脸检测置信度阈值
FACE_CONFIDENCE = 0.5

# 视频检测时的抽帧间隔
FRAME_INTERVAL = 20

# 推理设备：无 NVIDIA GPU 时使用 cpu；有可用 CUDA GPU 时可改为 "0"
DEVICE = "cpu"

# 支持的视频格式
VIDEO_FORMATS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)

# 自动创建目录
for path in [TEMP_DIR, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
