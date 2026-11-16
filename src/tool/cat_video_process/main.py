import os
import shutil
from tqdm import tqdm

from config import *
from modules.scene_detector import split_video_hsv, split_video_ssim
from modules.duration_filter import check_video_duration
from modules.cat_face_detector import CatFaceDetector


def main():

    detector = CatFaceDetector(
        MODEL_PATH,
        device=DEVICE,
        conf=FACE_CONFIDENCE
    )

    video_list = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VIDEO_FORMATS)
    ]
    print(f"Found {len(video_list)} videos to process.")

    for video_path in tqdm(video_list):

        filename = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        temp_output = os.path.join(TEMP_DIR, filename)

        os.makedirs(temp_output, exist_ok=True)

        # 1. 视频场景切分
        scenes = split_video_ssim(
            video_path,
            temp_output,
            SCENE_THRESHOLD
        )

        if not scenes:
            continue

        # 2. 获取切分后的视频
        clips = [
            os.path.join(temp_output, f)
            for f in os.listdir(temp_output)
            if f.lower().endswith(VIDEO_FORMATS)
        ]

        for clip in clips:

            # 3. 视频时长过滤
            if not check_video_duration(
                clip,
                MIN_DURATION
            ):
                os.remove(clip)
                continue

            # 4. 猫脸检测
            has_cat_face = detector.detect_video(
                clip,
                FRAME_INTERVAL
            )
            # print(f"Processing clip:{os.path.basename(clip)}, has_cat_face: {has_cat_face}")

            # 5. 保存有效视频
            if has_cat_face:

                save_path = os.path.join(
                    OUTPUT_DIR,
                    os.path.basename(clip)
                )
                print(f"has cat face, clip: {os.path.basename(clip)}")
                # print(f"save_path: {os.path.basename(save_path)}")

                os.replace(clip, save_path)

            # else:
            #     os.remove(clip)

        shutil.rmtree(temp_output)
        


if __name__ == "__main__":
    main()