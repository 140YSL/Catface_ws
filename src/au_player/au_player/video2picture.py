"""按 AU 顺序每隔 1.5 秒截帧，保存为 AU/AU-强度.png，强度保留一位小数。"""

import math
from pathlib import Path

import cv2


# 配置区：修改以下常量后直接运行，无需传入命令行参数。
VIDEO_PATH = '/home/ubuntu/Desktop/Catface_ws/cat_video/sim/simcat_au.mkv'
START_TIME = 8.0  # 开始时间：秒数，也支持 '00:01:12.5' 格式。
FRAME_INTERVAL = 1.51  # 截帧间隔，单位：秒。
OUTPUT_FOLDER_NAME = 'simcat_video_extra'  # 在视频所在目录下创建。


# 与 au_player.py 中 AU_DESCRIPTIONS 的顺序一致。
# 单独定义以便直接运行，不依赖 ROS 环境和已安装的关节映射文件。
AU_DESCRIPTIONS = {
    'LAU143': '左眼闭眼',
    'RAU143': '右眼闭眼',
    'AU109+110': '皱鼻+上唇提升',
    'AU118': '唇部前撅',
    'AU26': '下颌下垂',
    'LEAD101': '左耳向前',
    'LEAD104': '左耳旋转',
    'LEAD106': '左耳向后',
    'READ101': '右耳向前',
    'READ104': '右耳旋转',
    'READ106': '右耳向后',
}


def parse_start_time(value):
    """将秒数、MM:SS 或 HH:MM:SS 转为非负秒数，支持小数秒。"""
    try:
        parts = [float(part) for part in str(value).split(':')]
        if not 1 <= len(parts) <= 3:
            raise ValueError
        if any(not math.isfinite(part) or part < 0 for part in parts):
            raise ValueError
        if len(parts) > 1 and any(part >= 60 for part in parts[1:]):
            raise ValueError
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        if not math.isfinite(seconds):
            raise ValueError
        return seconds
    except ValueError as exc:
        raise ValueError(
            '开始时间应为非负秒数、MM:SS 或 HH:MM:SS，例如 12.5 或 00:01:12.5'
        ) from exc


def extract_frames(video_path, start_time):
    """从开始时间起连续取 286 个点，使用视频帧时间戳定位。"""
    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError('开始时间必须为非负有限秒数')
    if not math.isfinite(FRAME_INTERVAL) or FRAME_INTERVAL <= 0:
        raise ValueError('截帧间隔必须为正有限秒数')
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f'视频文件不存在：{video_path}')

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f'无法打开视频：{video_path}')
        output_dir = video_path.parent / OUTPUT_FOLDER_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for au_name in AU_DESCRIPTIONS:
            au_output_dir = output_dir / au_name
            au_output_dir.mkdir(parents=True, exist_ok=True)
            for strength_step in range(26):
                timestamp = start_time + count * FRAME_INTERVAL
                filename = f'{au_name}-{strength_step / 5:.1f}.png'
                if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
                    raise RuntimeError(f'视频不支持定位到 {timestamp:.3f} 秒')
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(
                        f'无法读取 {timestamp:.3f} 秒处的帧（{filename}），'
                        f'请检查视频时长；已保存 {count}/286 张到 {output_dir}'
                    )
                if not cv2.imwrite(str(au_output_dir / filename), frame):
                    raise RuntimeError(f'图片写入失败：{au_output_dir / filename}')
                count += 1
            print(f'{au_name}（{AU_DESCRIPTIONS[au_name]}）：已保存 26 张')
        return output_dir, count
    finally:
        capture.release()


def main():
    """使用脚本开头的配置截帧；同名 PNG 会被覆盖。"""
    try:
        output_dir, count = extract_frames(
            VIDEO_PATH, parse_start_time(START_TIME))
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise SystemExit(f'截帧失败：{exc}') from exc
    print(f'完成：共保存 {count} 张图片到 {output_dir}')


if __name__ == '__main__':
    main()
